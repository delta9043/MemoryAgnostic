import argparse
import json
import yaml
from data.locomo_loader import load_locomo10
from data.schema import ProcessedSample
from factory import build_chunker, build_pre_chunking_modules, build_memory_backend


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict):
    # 1. 데이터 로드
    dataset_cfg = cfg["dataset"]
    sample_idx = dataset_cfg.get("sample_idx", 0)
    if not isinstance(sample_idx, int):
        raise ValueError(f"sample_idx must be an integer, got {sample_idx!r}")

    raw = load_locomo10(dataset_cfg["path"], sample_idx=sample_idx)
    print(f"[loader] sample_id: {raw.sample_id}")
    print(f"[loader] num turns: {len(raw.turns)}")
    print(f"[loader] num qa: {len(raw.qa)}")

    # 2. 파이프라인 실행
    pre_chunking_modules = build_pre_chunking_modules(cfg)
    chunker = build_chunker(cfg)
    turns = raw.turns
    for module in pre_chunking_modules:
        turns = module.run(turns)
    print(f"[pipeline] turns after pre-chunking: {len(turns)}")

    chunks = chunker.chunk(turns)
    print(f"[chunker] num chunks: {len(chunks)}")

    # 3. ProcessedSample 생성
    module_names = [m["type"] for m in cfg["pipeline"]["pre_chunking"]]
    module_names.append(cfg["pipeline"]["chunker"]["type"])
    processed = ProcessedSample(
        sample_id=raw.sample_id,
        chunks=chunks,
        qa=raw.qa,
        metadata={"pipeline": module_names},
    )

    # 4. Memory Backend 연동
    backend = build_memory_backend(cfg)
    if backend is None:
        print("[memory] No memory backend configured. Skipping.")
        return

    print(f"[memory] Building memory from {len(chunks)} chunks...")
    backend.build(chunks)
    print("[memory] Memory build complete.")

    # 5. 평가
    eval_cfg = cfg.get("evaluation", {})
    result_file = eval_cfg.get("result_file", "results/output.json")

    results = []
    for i, qa in enumerate(processed.qa):
        question = qa.question
        answer_gt = qa.answer
        answer_pred = backend.query(question)
        results.append({
            "idx": i,
            "question": question,
            "answer_gt": answer_gt,
            "answer_pred": answer_pred,
        })
        if (i + 1) % 10 == 0:
            print(f"[eval] {i + 1}/{len(processed.qa)} questions answered")

    # 결과 저장
    import os
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[eval] Results saved to {result_file}")

    backend.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()