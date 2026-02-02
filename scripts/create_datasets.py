import json
from pathlib import Path
import re
import copy

from preprocessing.chunking import chunk_text_preserve_paragraphs
from preprocessing.fact_extraction import extract_atomic_facts_from_chunks

# SRC = "data/hotpot_train_v1.1.json" # big local file (keep ignored from git)
SRC = "../data/hotpot_train_v1.1.json"
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

D1_N = 100
D2_N = 20


def flip_one_word(sentence: str) -> tuple[str, str]:
    """
    Make a simple contradiction in a traceable way.
    Returns: (modified_sentence, rule_name)

    Priority:
      1) Flip a 4-digit year (e.g., 1844 -> 1944)
      2) Negate the first be-verb: is/are/was/were -> is not / ...
      3) Flip 'first' <-> 'last'
    """
    # 1) negate is/are/was/were (first occurrence)
    modified, n = re.subn(r"\b(is|are|was|were)\b", r"\1 not", sentence, count=1)
    if n > 0:
        return modified, "negate_be_verb"
    
    # 2) year 
    m = re.search(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-9]{2})\b", sentence)
    if m:
        year = int(m.group(0))
        # small shift only
        new_year = year + 10 if year <= 2015 else year - 10
        modified = sentence[:m.start()] + str(new_year) + sentence[m.end():]
        return modified, "flip_year(+/-10)"

    # 3) flipping first/last
    if re.search(r"\bfirst\b", sentence, flags=re.IGNORECASE):
        modified = re.sub(r"\bfirst\b", "last", sentence, flags=re.IGNORECASE, count=1)
        return modified, "flip_first_to_last"

    if re.search(r"\blast\b", sentence, flags=re.IGNORECASE):
        modified = re.sub(r"\blast\b", "first", sentence, flags=re.IGNORECASE, count=1)
        return modified, "flip_last_to_first"

    # fallback: append "not" (weak but ensures change)
    return sentence + " (not)", "fallback_append_not"


def flatten_context(context) -> str:
    paragraphs = []
    for title, paras in context:
        for p in paras:
            paragraphs.append(p)
    return "\n".join(paragraphs)


def build_atomic_facts_from_context(context_text: str):
    chunks = chunk_text_preserve_paragraphs(context_text, max_chars=1200)
    facts = extract_atomic_facts_from_chunks(chunks, min_len=10)
    return facts


def minimal_sample(ex):
    return {
        "id": ex["_id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "context": ex["context"],  # keeping og structure
    }


# loading full dataset 
with open(SRC, "r", encoding="utf-8") as f:
    full = json.load(f)

####### D1=  first 100 (clean)
d1 = [minimal_sample(ex) for ex in full[:D1_N]]
with open(OUT_DIR / "d1_100.json", "w", encoding="utf-8") as f:
    json.dump(d1, f, indent=2)
print("Saved D1:", len(d1), "samples -> data/d1_100.json")

##### D2 = first 20 with contradiction 
d2 = []
for ex in full[:D2_N]:
    ex_small = minimal_sample(ex)

    # extract atomic facts from OG context
    original_text = flatten_context(ex_small["context"])
    facts = build_atomic_facts_from_context(original_text)

    if len(facts) == 0:
        # If no facts, just keep sample
        ex_small["contradiction"] = {
            "status": "skipped_no_facts"
        }
        d2.append(ex_small)
        continue

    f0 = facts[0].text
    f0_mod, rule = flip_one_word(f0)

    # Inject the contradiction into the CONTEXT so your pipeline will see it.
    # We add it as an extra paragraph in the first context title.
    ctx = copy.deepcopy(ex_small["context"])
    if len(ctx) == 0:
        # create a dummy title if needed
        ctx = [["Injected", [f0, f0_mod]]]
    else:
        title0, paras0 = ctx[0]
        paras0 = list(paras0)
        # append original fact and contradiction explicitly (traceable)
        paras0.append(f0)
        paras0.append(f0_mod)
        ctx[0] = [title0, paras0]

    ex_small["context"] = ctx
    ex_small["contradiction"] = {
        "source_fact": f0,
        "modified_fact": f0_mod,
        "rule": rule,
        "injected_into": "context[0] paragraphs (appended f0 and f0_mod)"
    }

    d2.append(ex_small)

with open(OUT_DIR / "d2_20_contradictions.json", "w", encoding="utf-8") as f:
    json.dump(d2, f, indent=2)
print("Saved D2:", len(d2), "samples -> data/d2_20_contradictions.json")