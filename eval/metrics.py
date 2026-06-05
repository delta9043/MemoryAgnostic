"""
QA 평가 메트릭 계산 모듈.

SimpleMem/A-Mem 원본 평가 코드와 동일한 라이브러리/메트릭 사용:
- ROUGE (rouge_score)
- BLEU (nltk)
- BERTScore (bert_score)
- METEOR (nltk)
- Exact Match
- F1 (token-level)
"""

from collections import defaultdict
from typing import Dict, List

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# NLTK 데이터 다운로드 (한 번만)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


def calculate_rouge(prediction: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, prediction)

    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = prediction.lower().split()
    ref_tokens = [reference.lower().split()]
    smooth = SmoothingFunction().method1

    results = {}

    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))

        try:
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smooth,
            )
        except Exception:
            score = 0.0

        results[f"bleu{n}"] = score

    return results


def calculate_meteor(prediction: str, reference: str) -> float:
    try:
        return meteor_score(
            [reference.lower().split()],
            prediction.lower().split(),
        )
    except Exception:
        return 0.0


def calculate_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 (SQuAD-style).
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def calculate_bert_score(
    predictions: List[str],
    references: List[str],
) -> List[float]:
    """
    BERTScore는 배치로 계산하는 것이 효율적이라 별도 함수.
    """
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        verbose=False,
    )

    return f1.tolist()


def evaluate_results(
    results: List[dict],
    use_bertscore: bool = True,
) -> dict:
    """
    Args:
        results:
            [
                {
                    "answer_pred": "...",
                    "answer_gt": "...",
                    "category": "..."
                },
                ...
            ]

        use_bertscore:
            BERTScore 계산 여부.

    Returns:
        카테고리별 + overall 메트릭 dict.
    """

    if not results:
        return {}

    # BERTScore는 배치 계산
    bert_f1_list = None

    if use_bertscore:
        predictions = [r["answer_pred"] for r in results]
        references = [r["answer_gt"] for r in results]

        try:
            bert_f1_list = calculate_bert_score(predictions, references)
        except Exception as e:
            print(f"[metrics] BERTScore failed: {e}")
            bert_f1_list = [0.0] * len(results)

    # 개별 메트릭 계산
    per_item = []

    for i, result in enumerate(results):
        prediction = result["answer_pred"]
        ground_truth = result["answer_gt"]

        item = {
            "exact_match": calculate_exact_match(prediction, ground_truth),
            "f1": calculate_f1(prediction, ground_truth),
            "meteor": calculate_meteor(prediction, ground_truth),
            **calculate_rouge(prediction, ground_truth),
            **calculate_bleu(prediction, ground_truth),
        }

        if bert_f1_list is not None:
            item["bert_f1"] = bert_f1_list[i]

        per_item.append(
            {
                "category": result.get("category", "unknown"),
                **item,
            }
        )

    # 카테고리별 + overall 집계
    by_category = defaultdict(list)

    for item in per_item:
        category = item["category"]
        by_category[category].append(item)
        by_category["overall"].append(item)

    metric_keys = [key for key in per_item[0].keys() if key != "category"]

    aggregated = {}

    for category, items in by_category.items():
        aggregated[category] = {
            "count": len(items),
        }

        for key in metric_keys:
            avg = sum(item[key] for item in items) / len(items)
            aggregated[category][key] = round(avg * 100, 2)

    return aggregated


def print_metrics(metrics: dict) -> None:
    """
    metrics를 보기 좋게 출력.
    """
    category_order = [
        "single_hop",
        "temporal",
        "open_domain",
        "multi_hop",
        "adversarial",
        "overall",
    ]

    keys = [
        "exact_match",
        "f1",
        "rouge1_f",
        "rougeL_f",
        "bleu1",
        "bleu4",
        "meteor",
        "bert_f1",
    ]

    print("\n" + "=" * 100)
    print("[metrics] Evaluation Results")
    print("=" * 100)

    header = (
        f"{'category':<15} "
        + " ".join(f"{key:>10}" for key in keys)
        + f"  {'count':>6}"
    )

    print(header)
    print("-" * len(header))

    for category in category_order:
        if category not in metrics:
            continue

        metric = metrics[category]

        row = (
            f"{category:<15} "
            + " ".join(f"{metric.get(key, 0):>10.2f}" for key in keys)
            + f"  {metric['count']:>6}"
        )

        print(row)

    print("=" * 100)
