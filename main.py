### data loading 
from data.load_hotpot import load_first_hotpot_samples

train_path = "data/hotpot_train_v1.1.json"  # adjust your filename
samples = load_first_hotpot_samples(train_path, n=2)

for s in samples:
    print("=" * 60)
    print("ID:", s.sample_id)
    print("Question:", s.question)
    print("Gold answer:", s.answer)
    print("Full text length:", len(s.full_text))
    print("Text preview:", s.full_text[:300])

### chunking


from preprocessing.chunking import chunk_text_preserve_paragraphs

for s in samples:
    print("=" * 60)
    print("ID:", s.sample_id)
    print("Question:", s.question)
    print("Gold answer:", s.answer)
    print("Full text length:", len(s.full_text))
    print("Text preview:", s.full_text[:300])

    chunks = chunk_text_preserve_paragraphs(s.full_text, max_chars=1800)
    print(f"Num chunks: {len(chunks)}")
    for c in chunks[:2]:
        print(f"- Chunk {c.chunk_id} chars={len(c.text)} preview={c.text[:120]!r}")