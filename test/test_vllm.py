import time
import subprocess
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/delta9043/models/Qwen2.5-7B-Instruct"
PROMPT = "Tell me about the history of artificial intelligence in detail."
MAX_NEW_TOKENS = 200
N_RUNS = 3

def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    mem_mib = int(result.stdout.strip().split("\n")[0])
    return mem_mib / 1024

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=32768,
)

print(f"\n{'='*60}")
print(f"Model: {MODEL_PATH}")
print(f"{'='*60}")
print(f"GPU memory used: {get_gpu_memory():.2f} GB")  # 모델 로드 직후

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=MAX_NEW_TOKENS,
)

results = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    outputs = llm.generate([PROMPT], sampling_params)
    elapsed = time.perf_counter() - t0

    new_tokens = len(outputs[0].outputs[0].token_ids)
    tps = new_tokens / elapsed
    results.append(tps)
    print(f"  [Run {i}] new={new_tokens} | {elapsed:.2f}s | {tps:.2f} tok/s")

print(f"  → mean (전체): {sum(results)/len(results):.2f} tok/s")
print(f"  → mean (run 1+): {sum(results[1:])/len(results[1:]):.2f} tok/s")