import argparse
import copy
import json
import os
import time
import yaml

from data.locomo_loader import load_locomo10, load_filtered_json
from data.schema import ProcessedSample
from factory import build_chunker, build_pre_chunking_modules, build_memory_backend
from eval.metrics import evaluate_results, print_metrics


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict):
    # 0. vLLM URL 환경변수 오버라이드
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
        raw = load_filtered_json(
            filtered_path=dataset_cfg["path"],
            original_path=original_path,
            sample_idx=sample_idx,
        )
    else:
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

    module_names = [m["type"] for m in cfg["pipeline"]["pre_chunking"]]
    module_names.append(cfg["pipeline"]["chunker"]["type"])
    processed = ProcessedSample(
        sample_id=raw.sample_id,
        chunks=chunks,
        qa=raw.qa,
        metadata={"pipeline": module_names},
    )

    # 3. Memory Backend
    backend = build_memory_backend(cfg)
    if backend is None:
        print("[memory] No memory backend configured. Skipping.")
        return None

    print(f"[memory] Building memory from {len(chunks)} chunks...")
    build_start = time.time()
    backend.build(chunks)
    build_time = time.time() - build_start
    print(f"[memory] Memory build complete. ({build_time:.1f}s)")

    # 4. QA 수행
    eval_cfg = cfg.get("evaluation", {})
    result_file = eval_cfg.get("result_file", "results/output.json")
    total_qa = len(processed.qa)

    results = []
    qa_start = time.time()
    for i, qa in enumerate(processed.qa):
        question = qa.question
        answer_gt = qa.answer
        answer_pred = backend.query(question, qa.category, answer_gt)
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
    qa_time = time.time() - qa_start

    # 5. Evaluation
    print(f"\n[eval] Computing metrics for {len(results)} QA results...")
    eval_start = time.time()
    use_bertscore = eval_cfg.get("use_bertscore", True)
    metrics = evaluate_results(results, use_bertscore=use_bertscore)
    eval_time = time.time() - eval_start
    print_metrics(metrics)
    print(f"\n[eval] QA time: {qa_time:.1f}s, Metric time: {eval_time:.1f}s")

    # 6. 결과 저장
    output = {
        "sample_id": raw.sample_id,
        "pipeline": module_names,
        "results": results,
        "metrics": metrics,
        "build_time": round(build_time, 1),
        "qa_time": round(qa_time, 1),
        "eval_time": round(eval_time, 1),
    }
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[eval] Results saved to {result_file}")

    backend.reset()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    sample_idx = cfg["dataset"].get("sample_idx", None)

    if sample_idx is not None:
        # 단일 샘플 처리
        run(cfg)
    else:
        # 전체 샘플 순차 처리 후 결과 종합
        original_path = cfg["dataset"].get("original_path") or cfg["dataset"]["path"]
        with open(original_path, encoding="utf-8") as f:
            total = len(json.load(f))
        print(f"[main] Total samples: {total}")

        eval_cfg = cfg.get("evaluation", {})
        base_result_file = eval_cfg.get("result_file", "results/output.json")

        all_results = []
        all_metrics_by_cat = {}

        for i in range(total):
            print(f"\n[main] ===== Sample {i+1}/{total} =====")
            sample_cfg = copy.deepcopy(cfg)
            sample_cfg["dataset"]["sample_idx"] = i
            stem = base_result_file.replace(".json", "")
            sample_cfg["evaluation"]["result_file"] = f"{stem}_sample{i}.json"

            result = run(sample_cfg)
            if result:
                all_results.extend(result["results"])
                for cat, m in result["metrics"].items():
                    if cat not in all_metrics_by_cat:
                        all_metrics_by_cat[cat] = []
                    all_metrics_by_cat[cat].append(m)

        # 전체 메트릭 평균
        aggregated_metrics = {}
        for cat, metric_list in all_metrics_by_cat.items():
            keys = [k for k in metric_list[0].keys() if k != "count"]
            aggregated_metrics[cat] = {
                k: round(sum(m[k] for m in metric_list) / len(metric_list), 2)
                for k in keys
            }
            aggregated_metrics[cat]["count"] = sum(m["count"] for m in metric_list)

        # 종합 결과 저장
        output = {
            "total_samples": total,
            "pipeline": cfg["pipeline"],
            "metrics": aggregated_metrics,
            "results": all_results,
        }
        os.makedirs(os.path.dirname(base_result_file), exist_ok=True)
        with open(base_result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[main] All results saved to {base_result_file}")
        print_metrics(aggregated_metrics)


if __name__ == "__main__":
    main()