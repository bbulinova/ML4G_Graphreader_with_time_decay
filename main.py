import json
import argparse
from preprocessing.chunking import chunk_text_preserve_paragraphs
from preprocessing.fact_extraction import extract_atomic_facts_from_chunks
from preprocessing.temporal import assign_random_timestamps
from decay.time_decay import apply_time_decay
from query.retrieve import rank_facts_no_decay, rank_facts_with_decay
from evaluation.metrics import hit_at_k
from evaluation.qa_metrics import exact_match, f1_score, f1_star
from evaluation.llm_rater_proxy import final_llm_judgement
from graph.facts import FactNode
from graph.graph import FactGraph

#DATA_PATH = "data/hotpot_sample.json" ## jsut 2 samples loading

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    default="data/hotpot_sample.json",
    help="Path to dataset JSON file"
)
args = parser.parse_args()
DATA_PATH = args.data

with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} samples")

N = len(samples)
sum_f1_gplain = 0.0
sum_f1_gdecay = 0.0
sum_em_gplain = 0
sum_em_gdecay = 0

for idx, s in enumerate(samples):
    DEBUG = idx < 2
    question = s["question"]
    answer = s["answer"]

    # flatten context
    paragraphs = []
    for title, paras in s["context"]:
        for p in paras:
            paragraphs.append(p)

    full_text = "\n".join(paragraphs)

    if DEBUG:
        print("=" * 60)
        print("Question:", question)
        print("Answer:", answer)
        print("Text length:", len(full_text))
        print("Preview:", full_text[:200])

    # chunking
    chunks = chunk_text_preserve_paragraphs(full_text, max_chars=250)
    if DEBUG:
        print("Num chunks:", len(chunks))

    for c in chunks:
        preview = c.text[:80].replace("\n", " ")
        if DEBUG:
            print(f"  chunk {c.chunk_id} chars={len(c.text)} :: {preview!r}")
    
    # atomic facts
    facts = extract_atomic_facts_from_chunks(chunks)
    if DEBUG:
        print("Num atomic facts:", len(facts))
        for f in facts:
            print(f"  fact {f.fact_id} (chunk {f.chunk_id}): {f.text}")

    # adding time stamps to atomic facts
    #seed = abs(hash(s["_id"])) % (2**32) ## for smaples 2
    seed = abs(hash(s["question"])) % (2**32)
    temporal_facts = assign_random_timestamps(facts, seed=seed)
    if DEBUG:
        print("Temporal facts (first 5):")
        for tf in temporal_facts[:5]:
            print(f"  fact {tf.fact_id} (chunk {tf.chunk_id}) t={tf.timestamp}: {tf.text}")

    # time decay
    t_now = 10      # "current time" for this run
    lamb = 0.4      # decay rate (tune later)

    weighted_facts = apply_time_decay(temporal_facts, t_now=t_now, lamb=lamb)

    if DEBUG:
        print(f"Weighted facts (t_now={t_now}, lamb={lamb}) first 5:")
        for wf in weighted_facts[:5]:
            print(f"  fact {wf.fact_id} t={wf.timestamp} w={wf.weight:.3f}: {wf.text}")

    # sorting by weight 
    top_recent = sorted(weighted_facts, key=lambda x: x.weight, reverse=True)[:3]
    if DEBUG:
        print("Top 3 most 'recent' facts by weight:")
        for wf in top_recent:
            print(f"  t={wf.timestamp} w={wf.weight:.3f} :: {wf.text}")

    # retrieval 
    top_plain = rank_facts_no_decay(question, facts, top_k=5)
    top_decay = rank_facts_with_decay(question, weighted_facts, top_k=5)

    if DEBUG:
        print("\n--- RETRIEVAL (no decay) ---")
        for r in top_plain:
            print(f"score={r.score:.3f} :: {r.text}")

        print("\n--- RETRIEVAL (with time decay) ---")
        for r in top_decay:
            print(f"score={r.score:.3f} :: {r.text}")

    # ======================
    # STEP 3: GRAPH SCORING
    # ======================

    # --- build graph for plain retrieval ---
    plain_scores = {r.fact_id: r.score for r in top_plain}

    graph_plain = FactGraph()
    for f in facts:
        node = FactNode(
            fact_id=f.fact_id,
            text=f.text,
            chunk_id=f.chunk_id,
            timestamp=None,
            score=plain_scores.get(f.fact_id, 0.0)
        )
        graph_plain.add_node(node)

    graph_plain.build_edges_same_chunk()
    graph_plain.propagate(alpha=0.7)

    top_graph_plain = graph_plain.top_k(k=5)

    # --- build graph for time-decay retrieval ---
    decay_scores = {r.fact_id: r.score for r in top_decay}

    graph_decay = FactGraph()
    for f in weighted_facts:
        node = FactNode(
            fact_id=f.fact_id,
            text=f.text,
            chunk_id=f.chunk_id,
            timestamp=f.timestamp,
            score=decay_scores.get(f.fact_id, 0.0)
        )
        graph_decay.add_node(node)

    graph_decay.build_edges_same_chunk()
    graph_decay.propagate(alpha=0.7)

    top_graph_decay = graph_decay.top_k(k=5)

    #evaluation hit@k
    if DEBUG:
        for k in [1, 3, 5]:
            h_plain = hit_at_k(top_plain, answer, k)
            h_decay = hit_at_k(top_decay, answer, k)
            print(f"Hit@{k}  no-decay={h_plain}  with-decay={h_decay}")

    # paper evaluation metrics
    pred_plain = top_plain[0].text
    pred_decay = top_decay[0].text
    pred_graph_plain = top_graph_plain[0].text
    pred_graph_decay = top_graph_decay[0].text

    em_plain = exact_match(pred_plain, answer)
    f1_plain = f1_score(pred_plain, answer)
    f1s_plain = f1_star(pred_plain, answer)
    lr_plain = final_llm_judgement(pred_plain, answer)

    em_decay = exact_match(pred_decay, answer)
    f1_decay = f1_score(pred_decay, answer)
    f1s_decay = f1_star(pred_decay, answer)
    lr_decay = final_llm_judgement(pred_decay, answer)

    em_gplain = exact_match(pred_graph_plain, answer)
    f1_gplain = f1_score(pred_graph_plain, answer)
    f1s_gplain = f1_star(pred_graph_plain, answer)
    lr_gplain = final_llm_judgement(pred_graph_plain, answer)

    em_gdecay = exact_match(pred_graph_decay, answer)
    f1_gdecay = f1_score(pred_graph_decay, answer)
    f1s_gdecay = f1_star(pred_graph_decay, answer)
    lr_gdecay = final_llm_judgement(pred_graph_decay, answer)

    if DEBUG:
        print("METRICS (plain):",
        "EM", em_plain,
        "F1", round(f1_plain, 3),
        "F1*", round(f1s_plain, 3),
        "LR", lr_plain)

        print("METRICS (decay):",
            "EM", em_decay,
            "F1", round(f1_decay, 3),
            "F1*", round(f1s_decay, 3),
            "LR", lr_decay)

    if idx < 2:
        print("=" * 60)
        print("Question:", question)
        print("Answer:", answer)
        print("Pred (plain):", pred_plain)
        print("Pred (decay):", pred_decay)
        print("Pred (graph):", pred_graph_plain)
        print("Pred (graph+decay):", pred_graph_decay)
    
    sum_f1_gplain += f1_gplain
    sum_f1_gdecay += f1_gdecay
    sum_em_gplain += int(em_gplain)
    sum_em_gdecay += int(em_gdecay)


print("\n================ FINAL RESULTS ================")
print(f"Graph (no decay):   EM={sum_em_gplain/N:.3f}  F1={sum_f1_gplain/N:.3f}")
print(f"Graph + decay:      EM={sum_em_gdecay/N:.3f}  F1={sum_f1_gdecay/N:.3f}")