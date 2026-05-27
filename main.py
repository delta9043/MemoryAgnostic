import argparse
import json
import os
import time
import yaml
from collections import defaultdict

from data.locomo_loader import load_locomo10, load_filtered_json
from data.schema import ProcessedSample
from factory import build_chunker, build_pre_chunking_modules, build_memory_backend


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compute_f1(pred: str, gt: str) -> float:
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_bleu1(pred: str, gt: str) -> float:
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    gt_set = set(gt_tokens)
    matches = sum(1 for t in pred_tokens if t in gt_set)
    return matches / len(pred_tokens)


def _compute_metrics(results: list) -> dict:
    category_results = defaultdict(list)
    for r in results:
        cat = r.get("category", "unknown")
        f1 = _compute_f1(r["answer_pred"], r["answer_gt"])
        bleu1 = _compute_bleu1(r["answer_pred"], r["answer_gt"])
        category_results[cat].append({"f1": f1, "bleu1": bleu1})
        category_results["overall"].append({"f1": f1, "bleu1": bleu1})

    metrics = {}
    for cat, scores in category_results.items():
        metrics[cat] = {
            "f1": round(sum(s["f1"] for s in scores) / len(scores) * 100, 2),
            "bleu1": round(sum(s["bleu1"] for s in scores) / len(scores) * 100, 2),
            "count": len(scores),
        }
    return metrics


def run(cfg: dict):
    # 0. url 변경 확인
    base_url_override = os.environ.get("VLLM_BASE_URL")
    if base_url_override and "memory_backend" in cfg:
        cfg["memory_backend"]["base_url"] = base_url_override

    # 1. 데이터 로드
    dataset_cfg = cfg["dataset"]
    sample_idx = dataset_cfg.get("sample_idx", 0)
    if not isinstance(sample_idx, int):
        raise ValueError(f"sample_idx must be an integer, got {sample_idx!r}")

    original_path = dataset_cfg.get("original_path")
    if original_path:
        # filtered JSON 로드
        raw = load_filtered_json(
            filtered_path=dataset_cfg["path"],
            original_path=original_path,
            sample_idx=sample_idx,
        )
    else:
        # 원본 LoCoMo10 로드
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
    build_start = time.time()
    backend.build(chunks)
    build_time = time.time() - build_start
    print(f"[memory] Memory build complete. ({build_time:.1f}s)")

    # 5. 평가
    eval_cfg = cfg.get("evaluation", {})
    result_file = eval_cfg.get("result_file", "results/output.json")
    total_qa = len(processed.qa)

    results = []
    eval_start = time.time()
    for i, qa in enumerate(processed.qa):
        question = qa.question
        answer_gt = qa.answer
        answer_pred = backend.query(question, qa.category)
        results.append({
            "idx": i,
            "question": question,
            "answer_gt": answer_gt,
            "answer_pred": answer_pred,
            "category": qa.category,
        })
        print(f"[QA {i+1}/{total_qa}] ({qa.category}) Q: {question[:60]}")
        print(f"  GT:   {answer_gt[:80]}")
        print(f"  PRED: {answer_pred[:80]}")

    eval_time = time.time() - eval_start

    # 6. 메트릭 계산
    metrics = _compute_metrics(results)
    print("\n" + "=" * 60)
    print("[metrics] Results")
    print("=" * 60)
    cat_order = ["single_hop", "temporal", "open_domain", "multi_hop", "adversarial", "overall"]
    for cat in cat_order:
        if cat in metrics:
            m = metrics[cat]
            print(f"  {cat:<15} F1: {m['f1']:6.2f}  BLEU-1: {m['bleu1']:6.2f}  (n={m['count']})")
    print(f"\n[eval] Total QA time: {eval_time:.1f}s")

    # 7. 결과 저장
    output = {
        "sample_id": raw.sample_id,
        "pipeline": module_names,
        "results": results,
        "metrics": metrics,
        "build_time": round(build_time, 1),
        "eval_time": round(eval_time, 1),
    }
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
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