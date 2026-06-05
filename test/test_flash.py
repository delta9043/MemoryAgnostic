import time
import gc
import csv
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Config
# ============================================================

MODELS = [
    "/data/delta9043/models/Qwen2.5-7B-Instruct",
    "/data/delta9043/models/Qwen3-8B",
]

ATTN_IMPLS = [
    "flash_attention_2",
    "sdpa",
]

# FlashAttention-2 이점을 보려면 input length를 길게 잡아야 함
TARGET_INPUT_LENS = [
    1024,
    4096,
    8192,
    # 16384,  # RTX 3090에서는 OOM 가능성이 있으니 필요할 때만 열기
]

# batch size가 커질수록 FlashAttention-2가 유리해질 수 있음
BATCH_SIZES = [
    1,
    2,
    4,
    # 8,  # RTX 3090에서는 OOM 가능성이 있으니 필요할 때만 열기
]

# generate 전체 성능 측정용
MAX_NEW_TOKENS_LIST = [
    16,
    64,
    # 200,  # decode-heavy 상황도 보고 싶으면 열기
]

N_WARMUP = 1
N_RUNS = 3

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

OUTPUT_CSV = "attn_benchmark_results.csv"


# ============================================================
# Synthetic benchmark text
# ============================================================
# 긴 가사/논문/저작권 텍스트를 그대로 넣지 않고,
# benchmark 목적의 synthetic text를 반복해서 input length를 만든다.
#
# 주의:
# 이 코드는 task 성능이 아니라 attention backend 속도 비교가 목적이다.
# 따라서 의미 있는 프롬프트일 필요는 없고, 길이와 batch 조건을 통제하는 것이 중요하다.

SYNTHETIC_TEXT = """
This is a synthetic benchmark passage for measuring language model inference speed.
The passage contains neutral factual-style sentences without relying on copyrighted text.
The goal is to create a controllable prompt length for attention backend comparison.
The model should process this text as ordinary input while we measure prefill and decode speed.
Each repeated block increases the number of input tokens in a predictable way.
"""


# ============================================================
# Utility functions
# ============================================================

def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 80)
    print("CUDA information")
    print("=" * 80)
    print(f"torch version: {torch.__version__}")
    print(f"cuda version: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print("=" * 80)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)


def build_exact_length_inputs(tokenizer, target_input_len, batch_size):
    """
    정확히 target_input_len 길이의 input_ids를 만든다.

    여기서는 chat template을 쓰지 않는다.
    이유:
    - attention backend 성능 비교에서는 prompt 의미보다 길이 통제가 더 중요함
    - chat template을 사용한 뒤 truncation하면 generation prompt가 잘릴 수 있음
    - raw token ids로 테스트하면 input length를 정확히 맞출 수 있음

    실제 실험 프롬프트와 완전히 동일한 end-to-end 속도를 보고 싶다면,
    아래 함수를 chat template 기반으로 바꿔서 다시 측정하면 된다.
    """

    token_ids = tokenizer(
        SYNTHETIC_TEXT,
        add_special_tokens=False,
    )["input_ids"]

    if len(token_ids) == 0:
        raise ValueError("SYNTHETIC_TEXT produced empty token ids.")

    repeated = []
    while len(repeated) < target_input_len:
        repeated.extend(token_ids)

    repeated = repeated[:target_input_len]

    # bos token이 있으면 맨 앞에 넣어준다.
    # 단, 전체 길이는 target_input_len으로 유지한다.
    if tokenizer.bos_token_id is not None:
        repeated[0] = tokenizer.bos_token_id

    input_ids = torch.tensor(
        [repeated for _ in range(batch_size)],
        dtype=torch.long,
        device=DEVICE,
    )

    attention_mask = torch.ones_like(input_ids, device=DEVICE)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def load_model_and_tokenizer(model_path, attn_impl):
    print("\n" + "=" * 80)
    print(f"Loading model")
    print(f"model: {model_path}")
    print(f"requested attn_implementation: {attn_impl}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # decoder-only 모델에서 batch padding을 할 때 필요할 수 있음
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation=attn_impl,
    )

    model.eval()

    print(f"actual attn_implementation: {model.config._attn_implementation}")

    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3

    print(f"GPU memory allocated after load: {mem_allocated:.2f} GB")
    print(f"GPU memory reserved after load:   {mem_reserved:.2f} GB")

    return tokenizer, model


def measure_prefill(model, encoded, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """
    Prefill-only 측정.

    Prefill:
    - 입력 prompt 전체를 한 번에 모델에 통과시키는 단계
    - 긴 context에서 FlashAttention-2의 이점이 가장 잘 드러날 수 있음
    """

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    batch_size = input_ids.shape[0]
    input_len = input_ids.shape[1]
    total_input_tokens = batch_size * input_len

    # warmup
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        torch.cuda.synchronize()

    elapsed_list = []
    tps_list = []
    mem_list = []

    for run_idx in range(n_runs):
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tps = total_input_tokens / elapsed
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3

        elapsed_list.append(elapsed)
        tps_list.append(tps)
        mem_list.append(peak_mem)

        print(
            f"    [prefill run {run_idx}] "
            f"elapsed={elapsed:.4f}s | "
            f"tokens={total_input_tokens} | "
            f"tok/s={tps:.2f} | "
            f"peak_mem={peak_mem:.2f} GB"
        )

    return {
        "mode": "prefill",
        "mean_elapsed": sum(elapsed_list) / len(elapsed_list),
        "mean_tps": sum(tps_list) / len(tps_list),
        "mean_peak_mem_gb": sum(mem_list) / len(mem_list),
    }


def measure_generate(
    model,
    tokenizer,
    encoded,
    max_new_tokens,
    n_warmup=N_WARMUP,
    n_runs=N_RUNS,
):
    """
    Generate end-to-end 측정.

    Generate:
    - prefill + decode가 모두 포함됨
    - 실제 실험 wall-clock time과 더 가까움

    여기서는 min_new_tokens=max_new_tokens를 줘서
    EOS가 빨리 나와 측정이 흔들리는 것을 줄인다.
    """

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    batch_size = input_ids.shape[0]
    input_len = input_ids.shape[1]

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # warmup
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = model.generate(**generation_kwargs)
        torch.cuda.synchronize()

    elapsed_list = []
    tps_list = []
    mem_list = []

    for run_idx in range(n_runs):
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        output_len = output_ids.shape[1]
        actual_new_tokens_per_sample = output_len - input_len
        total_new_tokens = actual_new_tokens_per_sample * batch_size

        tps = total_new_tokens / elapsed
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3

        elapsed_list.append(elapsed)
        tps_list.append(tps)
        mem_list.append(peak_mem)

        print(
            f"    [generate run {run_idx}] "
            f"new={actual_new_tokens_per_sample} x batch={batch_size} | "
            f"elapsed={elapsed:.4f}s | "
            f"tok/s={tps:.2f} | "
            f"peak_mem={peak_mem:.2f} GB"
        )

    return {
        "mode": "generate",
        "mean_elapsed": sum(elapsed_list) / len(elapsed_list),
        "mean_tps": sum(tps_list) / len(tps_list),
        "mean_peak_mem_gb": sum(mem_list) / len(mem_list),
    }


def append_result_to_csv(path, row):
    fieldnames = [
        "model",
        "attn_impl",
        "actual_attn_impl",
        "mode",
        "target_input_len",
        "actual_input_len",
        "batch_size",
        "max_new_tokens",
        "mean_elapsed",
        "mean_tps",
        "mean_peak_mem_gb",
        "status",
        "error",
    ]

    file_exists = False
    try:
        with open(path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_one_model(model_path):
    for attn_impl in ATTN_IMPLS:
        tokenizer = None
        model = None

        try:
            cleanup()
            tokenizer, model = load_model_and_tokenizer(model_path, attn_impl)

            actual_attn_impl = model.config._attn_implementation

            for target_input_len in TARGET_INPUT_LENS:
                for batch_size in BATCH_SIZES:
                    print("\n" + "-" * 80)
                    print(
                        f"Case | model={model_path} | "
                        f"attn={attn_impl} | "
                        f"input_len={target_input_len} | "
                        f"batch={batch_size}"
                    )
                    print("-" * 80)

                    encoded = build_exact_length_inputs(
                        tokenizer=tokenizer,
                        target_input_len=target_input_len,
                        batch_size=batch_size,
                    )

                    actual_input_len = encoded["input_ids"].shape[1]

                    # ------------------------------------------------
                    # 1. Prefill-only benchmark
                    # ------------------------------------------------
                    try:
                        prefill_result = measure_prefill(
                            model=model,
                            encoded=encoded,
                        )

                        append_result_to_csv(
                            OUTPUT_CSV,
                            {
                                "model": model_path,
                                "attn_impl": attn_impl,
                                "actual_attn_impl": actual_attn_impl,
                                "mode": "prefill",
                                "target_input_len": target_input_len,
                                "actual_input_len": actual_input_len,
                                "batch_size": batch_size,
                                "max_new_tokens": "",
                                "mean_elapsed": prefill_result["mean_elapsed"],
                                "mean_tps": prefill_result["mean_tps"],
                                "mean_peak_mem_gb": prefill_result["mean_peak_mem_gb"],
                                "status": "ok",
                                "error": "",
                            },
                        )

                    except torch.cuda.OutOfMemoryError as e:
                        err = f"OOM in prefill: {repr(e)}"
                        print(f"    ERROR: {err}")
                        cleanup()

                        append_result_to_csv(
                            OUTPUT_CSV,
                            {
                                "model": model_path,
                                "attn_impl": attn_impl,
                                "actual_attn_impl": actual_attn_impl,
                                "mode": "prefill",
                                "target_input_len": target_input_len,
                                "actual_input_len": actual_input_len,
                                "batch_size": batch_size,
                                "max_new_tokens": "",
                                "mean_elapsed": "",
                                "mean_tps": "",
                                "mean_peak_mem_gb": "",
                                "status": "oom",
                                "error": err,
                            },
                        )

                        # 이 batch/input_len에서 OOM이면 generate도 거의 OOM 가능성이 큼
                        continue

                    # ------------------------------------------------
                    # 2. Generate benchmark
                    # ------------------------------------------------
                    for max_new_tokens in MAX_NEW_TOKENS_LIST:
                        print(
                            f"\n  Generate benchmark | "
                            f"max_new_tokens={max_new_tokens}"
                        )

                        try:
                            gen_result = measure_generate(
                                model=model,
                                tokenizer=tokenizer,
                                encoded=encoded,
                                max_new_tokens=max_new_tokens,
                            )

                            append_result_to_csv(
                                OUTPUT_CSV,
                                {
                                    "model": model_path,
                                    "attn_impl": attn_impl,
                                    "actual_attn_impl": actual_attn_impl,
                                    "mode": "generate",
                                    "target_input_len": target_input_len,
                                    "actual_input_len": actual_input_len,
                                    "batch_size": batch_size,
                                    "max_new_tokens": max_new_tokens,
                                    "mean_elapsed": gen_result["mean_elapsed"],
                                    "mean_tps": gen_result["mean_tps"],
                                    "mean_peak_mem_gb": gen_result["mean_peak_mem_gb"],
                                    "status": "ok",
                                    "error": "",
                                },
                            )

                        except torch.cuda.OutOfMemoryError as e:
                            err = f"OOM in generate: {repr(e)}"
                            print(f"    ERROR: {err}")
                            cleanup()

                            append_result_to_csv(
                                OUTPUT_CSV,
                                {
                                    "model": model_path,
                                    "attn_impl": attn_impl,
                                    "actual_attn_impl": actual_attn_impl,
                                    "mode": "generate",
                                    "target_input_len": target_input_len,
                                    "actual_input_len": actual_input_len,
                                    "batch_size": batch_size,
                                    "max_new_tokens": max_new_tokens,
                                    "mean_elapsed": "",
                                    "mean_tps": "",
                                    "mean_peak_mem_gb": "",
                                    "status": "oom",
                                    "error": err,
                                },
                            )

                        except Exception as e:
                            err = traceback.format_exc()
                            print(f"    ERROR in generate:\n{err}")
                            cleanup()

                            append_result_to_csv(
                                OUTPUT_CSV,
                                {
                                    "model": model_path,
                                    "attn_impl": attn_impl,
                                    "actual_attn_impl": actual_attn_impl,
                                    "mode": "generate",
                                    "target_input_len": target_input_len,
                                    "actual_input_len": actual_input_len,
                                    "batch_size": batch_size,
                                    "max_new_tokens": max_new_tokens,
                                    "mean_elapsed": "",
                                    "mean_tps": "",
                                    "mean_peak_mem_gb": "",
                                    "status": "error",
                                    "error": repr(e),
                                },
                            )

        except Exception as e:
            print("\n" + "!" * 80)
            print(f"FAILED to run model={model_path}, attn_impl={attn_impl}")
            print(traceback.format_exc())
            print("!" * 80)

        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup()


def main():
    check_cuda()

    print("\nBenchmark settings")
    print("=" * 80)
    print(f"MODELS: {MODELS}")
    print(f"ATTN_IMPLS: {ATTN_IMPLS}")
    print(f"TARGET_INPUT_LENS: {TARGET_INPUT_LENS}")
    print(f"BATCH_SIZES: {BATCH_SIZES}")
    print(f"MAX_NEW_TOKENS_LIST: {MAX_NEW_TOKENS_LIST}")
    print(f"N_WARMUP: {N_WARMUP}")
    print(f"N_RUNS: {N_RUNS}")
    print(f"OUTPUT_CSV: {OUTPUT_CSV}")
    print("=" * 80)

    for model_path in MODELS:
        run_one_model(model_path)

    print("\nDone.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()