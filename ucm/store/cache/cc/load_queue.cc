/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "load_queue.h"
#include <chrono>
#include <cstdint>
#include "logger/logger.h"
#include "thread/cpu_affinity.h"

namespace {

uint64_t NowUs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

size_t TensorBytesPerShard(const std::vector<size_t>& tensorSizes)
{
    size_t total = 0;
    for (const auto size : tensorSizes) { total += size; }
    return total;
}

}  // namespace

namespace UC::CacheStore {

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    streamNumber_ = config.streamNumber;
    useGdr_ = config.useGdr;
    cpuAffinityCores_ = config.cpuAffinityCores;
    transferProfile_ = {};
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, std::ref(started)};
    return fut.get();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    auto tp = waiter->startTp;
    auto tpWait = NowTime::Now();
    const auto nShard = task->desc.size();
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        ShardTask shardTask;
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index, true, true);
        shardTask.backendTaskHandle = 0;
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            Detail::TaskDesc backendTask{
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}}
            };
            backendTask.brief = "Backend2Cache";
            auto res = backend_->Load(std::move(backendTask));
            if (!res) [[unlikely]] {
                UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
        }
        shardTask.taskHandle = task->id;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        running_.Push(std::move(shardTask));
    }
    auto tpDispatch = NowTime::Now();
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream);
}

void LoadQueue::TransferOneTask(CopyStream& stream, ShardTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }
    const auto bytesPerShard = TensorBytesPerShard(tensorSizes_);
    auto logTransferProfile = [&](const char* status, uint64_t syncUs) {
        if (!transferProfile_.active || transferProfile_.shards == 0) { return; }
        const auto elapsedUs = NowUs() - transferProfile_.startUs;
        UC_INFO_UNLIMITED(
            "UCM transfer profile op=load direction=H2D task_id={} device_id={} use_gdr={} "
            "shards={} bytes={} submit_us={} sync_us={} elapsed_us={} status={}",
            transferProfile_.taskHandle, deviceId_, useGdr_ ? 1 : 0, transferProfile_.shards,
            transferProfile_.bytes, transferProfile_.submitUs, syncUs, elapsedUs, status);
        transferProfile_ = {};
    };
    auto s = Status::OK();
    do {
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] {
            logTransferProfile("error", 0);
            break;
        }
        if (!transferProfile_.active || transferProfile_.taskHandle != task.taskHandle) {
            transferProfile_ = {};
            transferProfile_.taskHandle = task.taskHandle;
            transferProfile_.startUs = NowUs();
            transferProfile_.active = true;
        }
        const auto submitStartUs = NowUs();
        s = HostToDeviceScatterAsync(stream.NextStream(), task.bufferHandle.Data(),
                                     task.shard.addrs.data());
        const auto submitUs = NowUs() - submitStartUs;
        if (s.Failure()) [[unlikely]] {
            logTransferProfile("error", 0);
            UC_ERROR("Failed({}) to do H2D batch async for task({}).", s, task.taskHandle);
            break;
        }
        transferProfile_.submitUs += submitUs;
        transferProfile_.shards += 1;
        transferProfile_.bytes += bytesPerShard;
        if (!task.waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        const auto syncStartUs = NowUs();
        s = stream.Synchronize();
        const auto syncUs = NowUs() - syncStartUs;
        holder_.clear();
        logTransferProfile(s.Success() ? "ok" : "error", syncUs);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, task.taskHandle);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(task.taskHandle); }
    if (task.waiter) { task.waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.taskHandle);
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    while (!task.bufferHandle.Ready()) {
        if (failureSet_->Contains(task.taskHandle)) { return Status::Error(); }
        std::this_thread::yield();
    }
    return Status::OK();
}

Status LoadQueue::HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                           void** device)
{
    const auto number = tensorSizes_.size();
    for (size_t i = 0, offset = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + offset);
        auto pDevice = device[i];
        auto size = tensorSizes_[i];
        auto s = stream->HostToDeviceAsync(pHost, pDevice, size);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D({}) batch({}/{}) async.", s, size, i, number);
            return s;
        }
        offset += size;
    }
    return Status::OK();
}

}  // namespace UC::CacheStore
