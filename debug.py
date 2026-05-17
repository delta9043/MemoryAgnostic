import json
import os
from dataclasses import asdict

from data.locomo_loader import load_locomo10_all
from core.chunker.fixed_size import FixedSizeChunker
from core.chunker.attention_similarity import AttentionSimilarityChunker
from core.filter.no_filter import NoFilter
from core.compressor.no_compressor import NoCompressor
from data.schema import ProcessedSample

LOCOMO_PATH = "/data/delta9043/datasets/locomo/locomo10.json"
OUTPUT_DIR = "debug_output"

LLMLINGUA_MODEL_PATH = "/data/delta9043/models/llmlingua-2"
EMBEDDER_MODEL_PATH = "/data/delta9043/models/all-MiniLM-L6-v2"


def run_pipeline(sample, pre_chunking_modules, chunker, pipeline_names):
    # 단일 샘플에 파이프라인 실행 후 ProcessedSample 반환
    turns = sample.turns
    for module in pre_chunking_modules:
        turns = module.run(turns)
    chunks = chunker.chunk(turns)
    return ProcessedSample(
        sample_id=sample.sample_id,
        chunks=chunks,
        qa=sample.qa,
        metadata={"pipeline": pipeline_names},
    )


def validate(processed):
    # 검증: 빈 chunks, 빈 text 확인
    warnings = []
    if not processed.chunks:
        warnings.append("no chunks produced")
    empty_text = [c.chunk_id for c in processed.chunks if not c.text.strip()]
    if empty_text:
        warnings.append(f"empty text in chunk_ids: {empty_text}")
    return warnings


def run_experiment(name, chunker, pipeline_names, samples):
    print(f"\n=== {name} ===\n")
    all_results = []

    for sample in samples:
        pre_chunking_modules = [NoFilter(), NoCompressor()]
        processed = run_pipeline(sample, pre_chunking_modules, chunker, pipeline_names)
        warnings = validate(processed)

        # chunk당 평균 turn 수
        avg_turns = sum(len(c.turns) for c in processed.chunks) / len(processed.chunks) if processed.chunks else 0

        print(
            f"sample_id={processed.sample_id} | "
            f"turns={len(sample.turns)} | "
            f"chunks={len(processed.chunks)} | "
            f"avg_turns_per_chunk={avg_turns:.1f} | "
            f"qa={len(processed.qa)}",
            end=""
        )
        if warnings:
            print(f" | WARNING: {warnings}")
        else:
            print(" | OK")

        all_results.append(asdict(processed))

    # JSON 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"locomo10_{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {output_path}")

    return all_results


def print_comparison(fixed_results, attn_results):
    print("\n=== 비교 요약 ===\n")
    print(f"{'sample_id':<12} {'fixed_chunks':>13} {'attn_chunks':>12} {'fixed_avg':>10} {'attn_avg':>10}")
    print("-" * 60)
    for f, a in zip(fixed_results, attn_results):
        fixed_chunks = len(f["chunks"])
        attn_chunks = len(a["chunks"])
        fixed_avg = sum(len(c["turns"]) for c in f["chunks"]) / fixed_chunks if fixed_chunks else 0
        attn_avg = sum(len(c["turns"]) for c in a["chunks"]) / attn_chunks if attn_chunks else 0
        print(
            f"{f['sample_id']:<12} "
            f"{fixed_chunks:>13} "
            f"{attn_chunks:>12} "
            f"{fixed_avg:>10.1f} "
            f"{attn_avg:>10.1f}"
        )


def main():
    print("데이터 로드 중...")
    samples = load_locomo10_all(LOCOMO_PATH)

    # FixedSizeChunker
    fixed_chunker = FixedSizeChunker(window=5, overlap=0)
    fixed_results = run_experiment(
        name="fixed_size",
        chunker=fixed_chunker,
        pipeline_names=["NoFilter", "NoCompressor", "FixedSizeChunker"],
        samples=samples,
    )

    # AttentionSimilarityChunker
    attn_chunker = AttentionSimilarityChunker(
        llmlingua_model_path=LLMLINGUA_MODEL_PATH,
        embedder_model_path=EMBEDDER_MODEL_PATH,
    )
    attn_results = run_experiment(
        name="attention_similarity",
        chunker=attn_chunker,
        pipeline_names=["NoFilter", "NoCompressor", "AttentionSimilarityChunker"],
        samples=samples,
    )

    print_comparison(fixed_results, attn_results)


if __name__ == "__main__":
    main()