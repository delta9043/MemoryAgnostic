import argparse
import copy
import json
import os
import sys
import time
import traceback
import yaml

from data.locomo_loader import load_locomo10, load_filtered_json
from data.schema import ProcessedSample
from factory import build_chunker, build_pre_chunking_modules, build_memory_backend
from eval.metrics import evaluate_results, print_metrics

# category 5(adversarial)의 두 번째 채점 기준. "거부"를 정답으로 보는 SimpleMem 방식
# (`SimpleMem/test_locomo10.py:871-874`). A-Mem 원본은 `adversarial_answer`(함정 오답
# 후보)를 GT로 쓰므로(`A-mem/test_advanced_robust.py:261`) 두 기준을 모두 채점한다.
CATEGORY5_GROUND_TRUTH_SIMPLEMEM = "Not mentioned in the conversation"


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _provenance(cfg: dict) -> dict:
    """결과 파일에 남길 설정 스냅샷.

    청커 파라미터(chunks_path, window/overlap)와 backend 파라미터(retrieve_k,
    top_k, reflection)가 수치를 좌우하므로 어떤 설정에서 나온 값인지 함께 저장한다.
    """
    backend = {k: v for k, v in (cfg.get("memory_backend") or {}).items()
               if k != "api_key"}
    return {"pipeline": cfg["pipeline"], "memory_backend": backend}


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
        # backend에는 qa.answer를 넘긴다(adversarial일 때 오답 후보로 쓰임).
        answer_pred = backend.query(question, qa.category, qa.answer)
        # answer_gt = A-Mem 방식(cat5는 loader가 이미 adversarial_answer를 담아준다).
        record = {
            "idx": i,
            "question": question,
            "answer_gt": qa.answer,
            "answer_pred": answer_pred,
            "category": qa.category,
        }
        if qa.category == "adversarial":
            record["answer_gt_simplemem"] = CATEGORY5_GROUND_TRUTH_SIMPLEMEM
        results.append(record)
        print(f"[QA {i+1}/{total_qa}] ({qa.category}) Q: {question[:60]}")
        print(f"  GT:   {qa.answer[:80]}")
        print(f"  PRED: {answer_pred[:80]}")
    qa_time = time.time() - qa_start

    # 5. Evaluation
    print(f"\n[eval] Computing metrics for {len(results)} QA results...")
    eval_start = time.time()
    use_bertscore = eval_cfg.get("use_bertscore", True)
    use_sbert = eval_cfg.get("use_sbert", True)
    metrics = evaluate_results(results, use_bertscore=use_bertscore, use_sbert=use_sbert)
    eval_time = time.time() - eval_start
    print_metrics(metrics)
    print(f"\n[eval] QA time: {qa_time:.1f}s, Metric time: {eval_time:.1f}s")

    # 6. 결과 저장
    output = {
        "sample_id": raw.sample_id,
        "pipeline": module_names,
        "config": _provenance(cfg),
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

        for i in range(total):
            print(f"\n[main] ===== Sample {i+1}/{total} =====")
            sample_cfg = copy.deepcopy(cfg)
            sample_cfg["dataset"]["sample_idx"] = i
            stem = base_result_file.replace(".json", "")
            sample_cfg["evaluation"]["result_file"] = f"{stem}_sample{i}.json"

            result = run(sample_cfg)
            if result:
                all_results.extend(result["results"])

        # 전체 집계: 모든 샘플의 질문을 풀링해 질문 단위로 평균(micro-average).
        # SimpleMem/A-Mem 원본(aggregate_metrics)과 동일 — 샘플 단순평균(macro)이 아니다.
        use_bertscore = cfg.get("evaluation", {}).get("use_bertscore", True)
        use_sbert = cfg.get("evaluation", {}).get("use_sbert", True)
        aggregated_metrics = evaluate_results(
            all_results, use_bertscore=use_bertscore, use_sbert=use_sbert,
        )

        # 종합 결과 저장
        output = {
            "total_samples": total,
            "pipeline": cfg["pipeline"],
            "config": _provenance(cfg),
            "metrics": aggregated_metrics,
            "results": all_results,
        }
        os.makedirs(os.path.dirname(base_result_file), exist_ok=True)
        with open(base_result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[main] All results saved to {base_result_file}")
        print_metrics(aggregated_metrics)


if __name__ == "__main__":
    # mem0 조건에서 main()이 끝나도 인터프리터가 종료되지 않아 잡이 GPU를 문 채 남았다.
    # 결과 파일은 이미 닫혔고 qdrant/sqlite는 매 실행 초기화라 종료 경로를 건너뛰어도 된다.
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)