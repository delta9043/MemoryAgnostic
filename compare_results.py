"""
compare_results.py — NoChunker(Qwen3-14B) vs LLMChunker 비교표 출력.

각 backend(A-Mem, SimpleMem)에 대해 LoCoMo10 5개 category의 F1 / BLEU-1을
baseline(NoChunker) 대비 LLMChunker 차이(%p)로 보여준다.
(발표자료 V2 표의 "행 차이 %p" 포맷, 1회 실험 기준)

main.py가 전체 샘플 실행 후 저장한 result JSON의 metrics를 읽는다.
metrics는 category -> {f1, bleu1, ...} 이며 값은 0~100 스케일이다.

4조건 실행이 모두 끝난 뒤 별도로 한 번 실행한다(개별 실행 스크립트와 분리).

사용법:
    python compare_results.py
    python compare_results.py --results_dir results
"""

import argparse
import json
import os

CATEGORY_ORDER = [
    "single_hop",
    "temporal",
    "open_domain",
    "multi_hop",
    "adversarial",
    "overall",
]

# (라벨, baseline 파일, llmchunk 파일)
BACKENDS = [
    ("A-Mem",     "amem_default.json",      "amem_llmchunker.json"),
    ("SimpleMem", "simplemem_default.json", "simplemem_llmchunker.json"),
]


def _load_metrics(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", {})


def _get(metrics, category, key):
    if not metrics or category not in metrics:
        return None
    return metrics[category].get(key)


def _fmt(v):
    return f"{v:6.2f}" if isinstance(v, (int, float)) else "   n/a"


def _fmt_delta(base, new):
    if not isinstance(base, (int, float)) or not isinstance(new, (int, float)):
        return "    n/a"
    d = new - base
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d):5.2f}"


def print_backend_table(label, base_metrics, llm_metrics):
    print("\n" + "=" * 92)
    print(f"[{label}]  NoChunker(Qwen3-14B)  vs  LLMChunker   — LoCoMo10")
    print("=" * 92)

    header = (
        f"{'category':<14}"
        f"{'F1_base':>9}{'F1_llm':>9}{'ΔF1(%p)':>10}   "
        f"{'BLEU1_base':>11}{'BLEU1_llm':>11}{'ΔBLEU1(%p)':>12}"
    )
    print(header)
    print("-" * len(header))

    for cat in CATEGORY_ORDER:
        f1_b = _get(base_metrics, cat, "f1")
        f1_l = _get(llm_metrics, cat, "f1")
        bl_b = _get(base_metrics, cat, "bleu1")
        bl_l = _get(llm_metrics, cat, "bleu1")
        if all(v is None for v in (f1_b, f1_l, bl_b, bl_l)):
            continue
        row = (
            f"{cat:<14}"
            f"{_fmt(f1_b):>9}{_fmt(f1_l):>9}{_fmt_delta(f1_b, f1_l):>10}   "
            f"{_fmt(bl_b):>11}{_fmt(bl_l):>11}{_fmt_delta(bl_b, bl_l):>12}"
        )
        print(row)

    print("=" * 92)
    print("ΔX(%p) = LLMChunker − NoChunker (양수면 LLMChunker가 향상)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    for label, base_file, llm_file in BACKENDS:
        base_metrics = _load_metrics(os.path.join(args.results_dir, base_file))
        llm_metrics = _load_metrics(os.path.join(args.results_dir, llm_file))
        if base_metrics is None and llm_metrics is None:
            print(f"\n[{label}] 결과 파일 없음 ({base_file}, {llm_file}) — 건너뜀")
            continue
        if base_metrics is None:
            print(f"\n[{label}] 경고: baseline 결과 없음 ({base_file})")
        if llm_metrics is None:
            print(f"\n[{label}] 경고: llmchunk 결과 없음 ({llm_file})")
        print_backend_table(label, base_metrics or {}, llm_metrics or {})


if __name__ == "__main__":
    main()
