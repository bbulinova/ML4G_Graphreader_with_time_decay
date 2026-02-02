import re
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())
    return text

#### EXACT MATCH
def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

#### F1 
def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

##### F1*
def f1_star(pred: str, gold: str, recall_threshold: float = 0.5) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    recall = num_same / len(gold_tokens)
    if recall < recall_threshold:
        return 0.0

    precision = num_same / len(pred_tokens)
    return 2 * precision * recall / (precision + recall)