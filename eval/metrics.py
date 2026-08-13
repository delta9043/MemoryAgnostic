"""
QA 평가 메트릭 계산 모듈.

채점 로직은 
A-Mem(`A-mem/utils.py:calculate_metrics`)과 
SimpleMem(`SimpleMem/test_locomo10.py:calculate_metrics`) 방식 차용
-> category 1/2/3/4는 채점 방식 동일

category5 (adversarial) 채점 방식
- A-Mem  : GT = `adversarial_answer`(함정 오답 후보)     | ref: `test_advanced_robust.py:261`
- SimpleMem: GT = "Not mentioned in the conversation"  | ref: `test_locomo10.py:871-874`
두 채점 방식 전부 산출 : adversarial_amem / adversarial_simplemem
"""

from collections import defaultdict
from typing import Dict, List, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# NLTK 데이터 확인. 리소스 이름을 대조하는 대신 실제로 쓰는 함수를 불러 본다
# (word_tokenize는 punkt가 아니라 punkt_tab, meteor는 omw까지 필요할 수 있다).
# 채점은 QA를 다 돈 뒤에 오므로 여기서 안 죽이면 몇 시간을 날린 뒤에 터진다.
def _check_nltk_data() -> None:
    def probe():
        nltk.word_tokenize("the quick brown fox")
        meteor_score([["the", "quick"]], ["the", "quick"])

    try:
        probe()
    except LookupError as first:
        for package in ("punkt_tab", "wordnet", "omw-1.4"):
            nltk.download(package, quiet=True)
        try:
            probe()
        except LookupError:
            raise RuntimeError(
                "NLTK 데이터 없음 — 오프라인 노드면 다운로드도 실패한다. "
                "NLTK_DATA를 데이터가 있는 경로로 지정할 것.\n"
                f"검색 경로: {nltk.data.path}"
            ) from first


_check_nltk_data()


# BLEU weight 목록 (A-Mem utils.py:55 | SimpleMem test_locomo10.py:287)
BLEU_WEIGHTS = [
    (1, 0, 0, 0),
    (0.5, 0.5, 0, 0),
    (0.33, 0.33, 0.33, 0),
    (0.25, 0.25, 0.25, 0.25),
]

# 전 메트릭 0 설정 (A-Mem utils.py:112-127 | SimpleMem test_locomo10.py:502-518)
ZERO_METRICS = {
    "exact_match": 0.0,
    "f1": 0.0,
    "rouge1_f": 0.0,
    "rouge2_f": 0.0,
    "rougeL_f": 0.0,
    "bleu1": 0.0,
    "bleu2": 0.0,
    "bleu3": 0.0,
    "bleu4": 0.0,
    "meteor": 0.0,
}


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
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    smooth = SmoothingFunction().method1

    results = {}

    for n, weights in enumerate(BLEU_WEIGHTS, start=1):
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
        return meteor_score([reference.split()], prediction.split())
    except Exception:
        return 0.0


def simple_tokenize(text: str) -> List[str]:
    # Convert to string if not already
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_f1(prediction: str, reference: str) -> float:
    """Token-level F1. SimpleMem/A-Mem 원본과 동일하게 set 기반(중복 토큰 무시)."""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        return 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1


def calculate_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def calculate_pair_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """예측-정답 한 쌍의 메트릭 계산"""
    if not prediction or not reference:
        return dict(ZERO_METRICS)

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    return {
        "exact_match": calculate_exact_match(prediction, reference),
        "f1": calculate_f1(prediction, reference),
        **calculate_rouge(prediction, reference),
        **calculate_bleu(prediction, reference),
        "meteor": calculate_meteor(prediction, reference),
    }


def calculate_bert_score(prediction: str, reference: str) -> Dict[str, float]:
    """BERTScore"""
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {
            "bert_precision": P.item(),
            "bert_recall": R.item(),
            "bert_f1": F1.item(),
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
        }


_sentence_model = None # Model
_sentence_model_loaded = False # Flag


def _get_sentence_model():
    """all-MiniLM-L6-v2를 로드 (원본은 import 시에 생성)"""
    global _sentence_model, _sentence_model_loaded

    if not _sentence_model_loaded:
        _sentence_model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer

            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer model: {e}")
            _sentence_model = None

    return _sentence_model


def calculate_sbert_similarity(prediction: str, reference: str) -> float:
    """SBERT 코사인 유사도"""
    model = _get_sentence_model()
    if model is None:
        return 0.0

    try:
        from sentence_transformers.util import pytorch_cos_sim

        embedding1 = model.encode([prediction], convert_to_tensor=True)
        embedding2 = model.encode([reference], convert_to_tensor=True)

        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0


# adversarial 채점 기준별 카테고리 라벨
ADV_LABEL_AMEM = "adversarial_amem"
ADV_LABEL_SIMPLEMEM = "adversarial_simplemem"

# 집계 라벨 = {합치는 방법}_{adversarial 처리}
# wmean = 문항 수로 가중한 평균(micro) / mean = 카테고리 균등 평균(macro)
WMEAN_ADV_AMEM = "wmean_adv-amem"
WMEAN_ADV_SIMPLEMEM = "wmean_adv-simplemem"
WMEAN_NO_ADV = "wmean_no-adv"
MEAN_NO_ADV = "mean_no-adv"

# mean_no-adv 대상 — SimpleMem 논문 Table 1/3의 Average와 같은 4개
MEAN_CATEGORIES = ["single_hop", "temporal", "open_domain", "multi_hop"]


def _scoring_units(results: List[dict]) -> List[dict]:
    """결과 항목을 '채점 단위'로 펼친다.

    adversarial은 기준이 두 개라 한 예측이 두 단위가 된다(라벨이 다름).
    나머지 카테고리는 1:1.
    """
    units = []

    for result in results:
        prediction = result.get("answer_pred") or ""
        category = result.get("category", "unknown")
        reference = result.get("answer_gt") or ""

        if category == "adversarial":
            gt_simplemem = result.get("answer_gt_simplemem")

            units.append({"label": ADV_LABEL_AMEM, "pred": prediction, "ref": reference})
            if gt_simplemem is not None:
                units.append({
                    "label": ADV_LABEL_SIMPLEMEM,
                    "pred": prediction,
                    "ref": gt_simplemem,
                })
        else:
            units.append({"label": category, "pred": prediction, "ref": reference})

    return units


def evaluate_results(
    results: List[dict],
    use_bertscore: bool = True,
    use_sbert: bool = True,
) -> dict:
    """
    Args:
        results:
            [
                {
                    "answer_pred": "...",
                    "answer_gt": "...",             # A-Mem 방식 GT (cat5 = adversarial_answer)
                    "answer_gt_simplemem": "...",   # adversarial만. 없으면 해당 라벨 생략
                    "category": "..."
                },
                ...
            ]

        use_bertscore / use_sbert:
            무거운 임베딩 지표 계산 여부.

    Returns:
        카테고리별 + 집계 메트릭 dict. 값은 ×100.

        카테고리 라벨:
        - single_hop / multi_hop / temporal / open_domain
        - adversarial_amem      : GT = adversarial_answer (A-Mem·LoCoMo 방식)
        - adversarial_simplemem : GT = "Not mentioned in the conversation" (SimpleMem 방식)

        집계 라벨 = {합치는 방법}_{adversarial 처리}:
        - wmean_adv-amem        : 비-adversarial + adversarial_amem, 문항 가중
        - wmean_adv-simplemem   : 비-adversarial + adversarial_simplemem, 문항 가중
        - wmean_no-adv          : 비-adversarial만, 문항 가중
        - mean_no-adv           : 비-adversarial만, 카테고리 균등
                                  (SimpleMem 논문 Table 1/3의 Average와 같은 방식)
    """

    if not results:
        return {}

    units = _scoring_units(results)

    per_unit = []

    for unit in units:
        item = calculate_pair_metrics(unit["pred"], unit["ref"])

        # 빈 예측/정답이면 임베딩 지표도 0 (원본과 동일)
        empty = not unit["pred"] or not unit["ref"]

        # 임베딩 지표도 원본과 같이 pair 단위로 계산한다
        if use_bertscore:
            item["bert_f1"] = (
                0.0 if empty
                else calculate_bert_score(unit["pred"], unit["ref"])["bert_f1"]
            )
        if use_sbert:
            item["sbert_similarity"] = (
                0.0 if empty
                else calculate_sbert_similarity(unit["pred"], unit["ref"])
            )

        per_unit.append({"label": unit["label"], **item})

    # 카테고리별 + wmean 집계 (문항 가중 평균)
    by_category = defaultdict(list)

    for item in per_unit:
        label = item["label"]
        by_category[label].append(item)

        if label == ADV_LABEL_AMEM:
            by_category[WMEAN_ADV_AMEM].append(item)
        elif label == ADV_LABEL_SIMPLEMEM:
            by_category[WMEAN_ADV_SIMPLEMEM].append(item)
        else:
            by_category[WMEAN_ADV_AMEM].append(item)
            by_category[WMEAN_ADV_SIMPLEMEM].append(item)
            by_category[WMEAN_NO_ADV].append(item)

    # adversarial_simplemem이 없는 결과(구 파일)면 wmean_adv-simplemem은 의미가 없다
    if ADV_LABEL_SIMPLEMEM not in by_category:
        by_category.pop(WMEAN_ADV_SIMPLEMEM, None)

    metric_keys = [key for key in per_unit[0].keys() if key != "label"]

    aggregated = {}

    for category, items in by_category.items():
        aggregated[category] = {
            "count": len(items),
        }

        for key in metric_keys:
            avg = sum(item[key] for item in items) / len(items)
            aggregated[category][key] = round(avg * 100, 2)

    # mean_no-adv = 카테고리 4개 점수의 균등 평균. 논문이 표에 찍힌 값을 평균했으므로
    # 원시값이 아니라 이미 반올림된 카테고리 값을 평균해야 산술이 일치한다.
    # 4개가 다 있을 때만 만든다 — 3개짜리 평균이 논문의 4개짜리와 비교되면 조용히 틀린다.
    if all(category in aggregated for category in MEAN_CATEGORIES):
        rows = [aggregated[category] for category in MEAN_CATEGORIES]
        aggregated[MEAN_NO_ADV] = {"count": sum(row["count"] for row in rows)}

        for key in metric_keys:
            aggregated[MEAN_NO_ADV][key] = round(
                sum(row[key] for row in rows) / len(rows), 2
            )

    return aggregated


CATEGORY_ORDER = [
    "single_hop",
    "temporal",
    "open_domain",
    "multi_hop",
    ADV_LABEL_AMEM,
    ADV_LABEL_SIMPLEMEM,
    WMEAN_ADV_AMEM,
    WMEAN_ADV_SIMPLEMEM,
    WMEAN_NO_ADV,
    MEAN_NO_ADV,
]


def print_metrics(metrics: dict) -> None:
    """
    metrics를 보기 좋게 출력.
    """
    keys = [
        "exact_match",
        "f1",
        "rouge1_f",
        "rougeL_f",
        "bleu1",
        "bleu4",
        "meteor",
        "bert_f1",
        "sbert_similarity",
    ]

    print("\n" + "=" * 120)
    print("[metrics] Evaluation Results")
    print("=" * 120)

    header = (
        f"{'category':<22} "
        + " ".join(f"{key:>10}" for key in keys)
        + f"  {'count':>6}"
    )

    print(header)
    print("-" * len(header))

    for category in CATEGORY_ORDER:
        if category not in metrics:
            continue

        metric = metrics[category]

        row = (
            f"{category:<22} "
            + " ".join(f"{metric.get(key, 0):>10.2f}" for key in keys)
            + f"  {metric['count']:>6}"
        )

        print(row)

    print("=" * 120)
    print("adversarial_amem      = GT: adversarial_answer (A-Mem/LoCoMo 방식)")
    print("adversarial_simplemem = GT: \"Not mentioned in the conversation\" (SimpleMem 방식)")
    print("wmean_* = 문항 가중 평균 | mean_* = 카테고리 균등 평균(논문 Average 방식)")
