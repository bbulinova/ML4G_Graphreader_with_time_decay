import json
from preprocessing.chunking import chunk_text_preserve_paragraphs
from preprocessing.fact_extraction import extract_atomic_facts_from_chunks
from preprocessing.temporal import assign_random_timestamps
from decay.time_decay import apply_time_decay

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

    # adding time stamps to atomic facts
    temporal_facts = assign_random_timestamps(facts, seed=42)

    print("Temporal facts (first 5):")
    for tf in temporal_facts[:5]:
        print(f"  fact {tf.fact_id} (chunk {tf.chunk_id}) t={tf.timestamp}: {tf.text}")

    # time decay
    t_now = 10      # "current time" for this run
    lamb = 0.4      # decay rate (tune later)

    weighted_facts = apply_time_decay(temporal_facts, t_now=t_now, lamb=lamb)

    print(f"Weighted facts (t_now={t_now}, lamb={lamb}) first 5:")
    for wf in weighted_facts[:5]:
        print(f"  fact {wf.fact_id} t={wf.timestamp} w={wf.weight:.3f}: {wf.text}")

    # sorting by weight 
    top_recent = sorted(weighted_facts, key=lambda x: x.weight, reverse=True)[:3]
    print("Top 3 most 'recent' facts by weight:")
    for wf in top_recent:
        print(f"  t={wf.timestamp} w={wf.weight:.3f} :: {wf.text}")