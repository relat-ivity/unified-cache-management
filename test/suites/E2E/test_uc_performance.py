import dataclasses
import json
import os
import shutil
import subprocess

import pytest
import requests
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.llmperf.run_inference import inference_results
from common.uc_eval.task import (
    DocQaPerfTask,
    MultiTurnDialogPerfTask,
    SyntheticPerfTask,
)
from common.uc_eval.utils.data_class import ModelConfig, PerfConfig

RUN_LEGACY_UC_PERF = os.getenv("RUN_LEGACY_UC_PERF", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LEGACY_UC_PERF_SKIP = pytest.mark.skipif(
    not RUN_LEGACY_UC_PERF,
    reason="Set RUN_LEGACY_UC_PERF=1 to run legacy llmperf/uc_eval cases",
)

VLLM_BENCH_TIMEOUT_S = int(os.getenv("VLLM_BENCH_TIMEOUT_S", "21600"))


def _vllm_bench_command(num_prompts: int, prefix_len: int, suffix_len: int) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        "http://127.0.0.1:8000",
        "--endpoint",
        "/v1/completions",
        "--model",
        "/models/MiniMax-M2.7",
        "--dataset-name",
        "prefix_repetition",
        "--num-prompts",
        str(num_prompts),
        "--prefix-repetition-prefix-len",
        str(prefix_len),
        "--prefix-repetition-suffix-len",
        str(suffix_len),
        "--prefix-repetition-num-prefixes",
        "1",
        "--prefix-repetition-output-len",
        "2",
        "--max-concurrency",
        "1",
        "--seed",
        "90",
    ]


def _run_vllm_bench_command(phase: str, command: list[str]) -> None:
    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        pytest.skip("vllm CLI not found")

    runnable = [vllm_bin, *command[1:]]
    print(f"\n[INFO] Running vLLM bench phase: {phase}")
    print("[INFO] Command: " + " ".join(command))

    try:
        proc = subprocess.run(
            runnable,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=VLLM_BENCH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        print(output)
        pytest.fail(
            f"vllm bench {phase} timed out after {VLLM_BENCH_TIMEOUT_S} seconds"
        )

    print(proc.stdout)
    assert proc.returncode == 0, f"vllm bench {phase} failed"


@pytest.mark.stage(2)
@pytest.mark.feature("uc_performance_test")
def test_vllm_bench_90_reuse_rate():
    _run_vllm_bench_command(
        "prefill_90",
        _vllm_bench_command(num_prompts=1, prefix_len=115200, suffix_len=0),
    )
    _run_vllm_bench_command(
        "bench_90",
        _vllm_bench_command(num_prompts=50, prefix_len=115200, suffix_len=12800),
    )


perf_scenarios = [
    # (mean_in, mean_out, max_req, concurrent, random_seed, hit_rate)
    (1000, 1024, 8, 8, 0, 0),
    (4000, 500, 1, 1, 0, 0),
]

perf_test_case_str = os.getenv("PERF_TEST_CASE") if RUN_LEGACY_UC_PERF else None
if not RUN_LEGACY_UC_PERF:
    pass
elif perf_test_case_str:
    try:
        parsed = json.loads(perf_test_case_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid = True
            result = []
            for item in parsed:
                if not isinstance(item, (list, tuple)) or len(item) != 6:
                    valid = False
                    break
                try:
                    result.append(tuple(int(x) for x in item))
                except (ValueError, TypeError):
                    valid = False
                    break
            if valid:
                perf_scenarios = result
                print(
                    f"Successfully loaded configuration from environment variable: {perf_scenarios}"
                )
            else:
                print(
                    "Environment variable format is invalid, using default configuration"
                )
        else:
            print("Parsed result is empty or not a list, using default configuration")
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}, using default configuration")
    except Exception as e:
        print(f"Unexpected parsing error: {e}, using default configuration")
else:
    print("PERF_TEST_CASE environment variable is not set, using default configuration")

if RUN_LEGACY_UC_PERF:
    print(f"Final perf_scenarios: {perf_scenarios}")


scenario_ids = [f"in_{s[0]}-out_{s[1]}-con_{s[3]}" for s in perf_scenarios]
TOTAL_COUNTER = len(perf_scenarios)
ROUND_COUNTER = 1
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "").lower() == "true"


def _send_profile_request(action: str) -> bool:
    """Send profiling request to vLLM server with blocking wait.

    Args:
        action: 'start' or 'stop'
    """
    server_url = config_instance.get_nested_config("llm_connection.server_url", "")
    if not server_url:
        raise RuntimeError("server_url not configured, cannot start profiling")

    url = f"{server_url}/{action}_profile"
    print(f"[INFO] Waiting For Profiler to {action}")
    resp = requests.post(url, timeout=1200)
    resp.raise_for_status()
    print(f"[INFO] Profiler {action}ed successfully")


@LEGACY_UC_PERF_SKIP
@pytest.mark.stage(2)
@pytest.mark.feature("uc_performance_test")
@pytest.mark.parametrize(
    "in_tokens, out_tokens, max_req, concurrent, random_seed, hit_rate",
    perf_scenarios,
    ids=scenario_ids,
)
@export_vars
def test_performance(in_tokens, out_tokens, max_req, concurrent, random_seed, hit_rate):
    global TOTAL_COUNTER
    global ROUND_COUNTER

    if ENABLE_PROFILING:
        _send_profile_request("start")

    try:
        summary = inference_results(
            [in_tokens],
            [out_tokens],
            [max_req],
            [concurrent],
            [random_seed],
            [hit_rate],
            TOTAL_COUNTER,
            ROUND_COUNTER,
        )
    finally:
        if ENABLE_PROFILING:
            _send_profile_request("stop")

    ROUND_COUNTER += 1
    results = summary.get("results", {})

    # 构造扁平化的结果字典，方便后续分析和看板展示
    metrics = {
        # 输入指标
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "concurrent": concurrent,
        "sum_requests": max_req,
        "hit_rate": hit_rate,
        "mean_input_tokens": summary.get("mean_input_tokens"),
        "mean_output_tokens": summary.get("mean_output_tokens"),
        "ttft_mean_s": results.get("ttft_s", {}).get("mean"),
        "tpot_mean_s": results.get("tpot_s", {}).get("mean"),
        "e2e_mean_s": results.get("end_to_end_latency_s", {}).get("mean"),
        "elapsed_time_s": summary.get("elapsed_time"),
        "total_throughput": summary.get("total_throughput"),
        "incremental_throughput": summary.get("incremental_throughput"),
        "extra_info": os.getenv("TEST_EXTRA_INFO")
        or config_instance.get_nested_config("llm_connection.extra_info"),
    }

    for key, val in metrics.items():
        assert val is not None, f"Metric '{key}' is missing"

    return {"_name": "llmperf_result", "_proj": metrics}


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    cfg = config_instance.get_config("models") or {}
    field_name = [field.name for field in dataclasses.fields(ModelConfig)]
    kwargs = {k: v for k, v in cfg.items() if k in field_name and v is not None}
    return ModelConfig(**kwargs)


sync_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=False,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            benchmark_mode="default-perf",
            kv_hit_type="HBM",
            epoch_num=5,
            test_name="no gsa and no prefix cache",
        ),
    ),
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=True,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            prefix_cache_num=[0.8, 0.8],
            benchmark_mode="default-perf",
            kv_hit_type="HBM",  # HBM or DISK
            epoch_num=5,
            test_name="no gsa and enable prefix cache",
        ),
    ),
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=True,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            prefix_cache_num=[0.8, 0.8],
            benchmark_mode="stable-perf",
            kv_hit_type="HBM",
            epoch_num=5,
            test_name="no gsa and enable prefix cache and stable perf",
        ),
    ),
]


@LEGACY_UC_PERF_SKIP
@pytest.mark.feature("sync_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", sync_perf_cases)
@export_vars
def test_sync_perf(perf_config: PerfConfig, model_config: ModelConfig):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = SyntheticPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": perf_config.test_name, "_proj": result}


multiturn_dialogue_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="multi_turn_dialogue",
            dataset_file_path="datasets/multi_turn_dialogues/multiturndialog.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
            test_name="shartgpt and no prefix cache",
        ),
    )
]


@LEGACY_UC_PERF_SKIP
@pytest.mark.feature("dialogue_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", multiturn_dialogue_perf_cases)
@export_vars
def test_multiturn_dialogue_perf(perf_config: PerfConfig, model_config: ModelConfig):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = MultiTurnDialogPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": perf_config.test_name, "_data": result}


doc_qa_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
            test_name="longbench and no prefix cache",
        ),
    )
]


@LEGACY_UC_PERF_SKIP
@pytest.mark.feature("qa_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", doc_qa_perf_cases)
@export_vars
def test_doc_qa_perf(perf_config: PerfConfig, model_config: ModelConfig):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": perf_config.test_name, "_data": result}
