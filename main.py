# main.py
import json
from preprocessing.chunking import chunk_text_preserve_paragraphs
from preprocessing.fact_extraction import extract_atomic_facts_from_chunks

DATA_PATH = "data/hotpot_sample.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

print(f"Loaded {len(samples)} samples")

for s in samples:
    question = s["question"]
    answer = s["answer"]

    # flatten context
    paragraphs = []
    for title, paras in s["context"]:
        for p in paras:
            paragraphs.append(p)

    full_text = "\n".join(paragraphs)

    print("=" * 60)
    print("Question:", question)
    print("Answer:", answer)
    print("Text length:", len(full_text))
    print("Preview:", full_text[:200])

    # chunking
    chunks = chunk_text_preserve_paragraphs(full_text, max_chars=250)
    print("Num chunks:", len(chunks))

    for c in chunks:
        preview = c.text[:80].replace("\n", " ")
        print(f"  chunk {c.chunk_id} chars={len(c.text)} :: {preview!r}")
    
    # atomic facts
    facts = extract_atomic_facts_from_chunks(chunks)
    print("Num atomic facts:", len(facts))
    for f in facts:
        print(f"  fact {f.fact_id} (chunk {f.chunk_id}): {f.text}")