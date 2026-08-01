"""
rescore.py — 저장된 result.json을 LLM 재실행 없이 다시 채점한다.

사용법:
    python eval/rescore.py results/amem_default_sample0.json --no-bertscore --no-sbert
    python eval/rescore.py "results/**/result.json" --dataset data/locomo10.json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import (  # noqa: E402
    ADV_LABEL_AMEM,
    ADV_LABEL_SIMPLEMEM,
    CATEGORY_ORDER,
    evaluate_results,
    print_metrics,
)

CATEGORY5_GROUND_TRUTH_SIMPLEMEM = "Not mentioned in the conversation"

DATASET_CANDIDATES = [
    "data/locomo10.json",
    "../A-mem/data/locomo10.json",
    "/data/delta9043/datasets/locomo/locomo10.json",
]


def find_dataset(explicit: str = None) -> str:
    if explicit:
        return explicit
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "locomo10.json을 찾지 못했다. --dataset으로 경로를 지정할 것. "
        f"(찾아본 곳: {DATASET_CANDIDATES})"
    )


def load_adversarial_answers(dataset_path: str) -> dict:
    """question -> adversarial_answer 맵."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for sample in data:
        for qa in sample["qa"]:
            if qa.get("category") == 5:
                mapping[qa["question"]] = qa.get("adversarial_answer", "")
    return mapping


def normalize_rows(results: list, adv_answers: dict) -> int:
    """
    result 행에 두 기준의 GT를 채운다. 
    반환값 = 복원 실패한 cat5 행 수.
    """
    missing = 0

    for row in results:
        if row.get("category") != "adversarial":
            continue

        recovered = adv_answers.get(row.get("question"))
        if recovered is None:
            missing += 1
        else:
            row["answer_gt"] = recovered

        row["answer_gt_simplemem"] = CATEGORY5_GROUND_TRUTH_SIMPLEMEM

    return missing


def print_delta(old: dict, new: dict, keys=("f1", "bleu1")) -> None:
    """
    구 metrics 대비 Δ. 
    라벨이 바뀐 adversarial은 구 'adversarial'과 나란히 보여준다.
    """
    print("\n" + "-" * 78)
    print("[rescore] 구 metrics 대비 변화")
    print("-" * 78)

    header = f"{'category':<22}" + "".join(
        f"{k + '_old':>12}{k + '_new':>12}{'Δ':>9}" for k in keys
    )
    print(header)
    print("-" * len(header))

    for category in CATEGORY_ORDER:
        if category not in new:
            continue

        # 라벨이 바뀐 adversarial 두 줄은 구 'adversarial' 하나와 대조한다
        old_key = category
        if category in (ADV_LABEL_AMEM, ADV_LABEL_SIMPLEMEM) and category not in old:
            old_key = "adversarial"

        row = f"{category:<22}"
        for key in keys:
            new_val = new[category].get(key)
            old_val = old.get(old_key, {}).get(key)
            row += f"{old_val:>12.2f}" if old_val is not None else f"{'n/a':>12}"
            row += f"{new_val:>12.2f}" if new_val is not None else f"{'n/a':>12}"
            if old_val is None or new_val is None:
                row += f"{'n/a':>9}"
            else:
                row += f"{new_val - old_val:>+9.2f}"
        print(row)

    print("-" * 78)
    print("주의: adversarial 두 줄의 old는 구 'adversarial'(단일 기준) 값이다.")


def rescore_file(path: str, adv_answers: dict, use_bertscore: bool,
                 use_sbert: bool, suffix: str, write: bool) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results")
    if not results:
        print(f"[rescore] {path}: results 없음 — 건너뜀")
        return

    print(f"\n{'=' * 78}\n[rescore] {path}  ({len(results)} QA)\n{'=' * 78}")

    missing = normalize_rows(results, adv_answers)
    if missing:
        print(f"[rescore] 경고: adversarial_answer 복원 실패 {missing}건 "
              "(저장된 answer_gt를 그대로 씀)")

    old_metrics = data.get("metrics", {})
    new_metrics = evaluate_results(
        results, use_bertscore=use_bertscore, use_sbert=use_sbert,
    )

    print_metrics(new_metrics)
    print_delta(old_metrics, new_metrics)

    if not write:
        return

    data["metrics"] = new_metrics
    data["metrics_previous"] = old_metrics
    out_path = path.replace(".json", f"{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[rescore] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="result json 경로 또는 glob 패턴")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--suffix", type=str, default="_rescored")
    parser.add_argument("--dry-run", action="store_true",
                        help="파일을 쓰지 않고 표만 출력")
    args = parser.parse_args()

    dataset_path = find_dataset(args.dataset)
    print(f"[rescore] dataset: {dataset_path}")
    adv_answers = load_adversarial_answers(dataset_path)
    print(f"[rescore] adversarial 정답 맵: {len(adv_answers)}개")

    files = []
    for pattern in args.paths:
        matched = sorted(glob.glob(pattern, recursive=True))
        files.extend(matched if matched else [pattern])

    for path in files:
        if not os.path.exists(path):
            print(f"[rescore] {path}: 파일 없음 — 건너뜀")
            continue
        if args.suffix in os.path.basename(path):
            print(f"[rescore] {path}: 이미 재채점된 파일 — 건너뜀")
            continue
        rescore_file(
            path, adv_answers,
            use_bertscore=not args.no_bertscore,
            use_sbert=not args.no_sbert,
            suffix=args.suffix,
            write=not args.dry_run,
        )


if __name__ == "__main__":
    main()
