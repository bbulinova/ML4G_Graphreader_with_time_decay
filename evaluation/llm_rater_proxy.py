from evaluation.qa_metrics import exact_match, f1_score


def llm_rater_strict(pred: str, gold: str) -> str:
    if exact_match(pred, gold) or f1_score(pred, gold) >= 0.9:
        return "Yes"
    return "No"


def llm_rater_lenient(pred: str, gold: str) -> str:
    f1 = f1_score(pred, gold)
    if f1 >= 0.3:
        return "Yes, partially"
    return "No"


def final_llm_judgement(pred: str, gold: str) -> str:
    strict = llm_rater_strict(pred, gold)
    lenient = llm_rater_lenient(pred, gold)

    if strict == "Yes":
        return "Correct"
    if strict == "No" and lenient == "Yes, partially":
        return "Partially correct"
    return "Incorrect"