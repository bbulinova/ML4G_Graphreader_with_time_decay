# GraphReader with Time Decay

This repository accompanies a course project on reproducing and extending
GraphReader-style graph-based retrieval for long-context question answering.

The project reimplements a lightweight version of the GraphReader pipeline and
introduces a temporal decay mechanism to model recency effects in fact retrieval.

---

## Project Overview

Implemented components:
- Paragraph-preserving chunking
- Atomic fact extraction
- Graph construction over atomic facts
- Graph-based score propagation
- Temporal decay weighting for recency-aware retrieval
- Evaluation with EM, F1, F1*, Hit@K, and LLM-based proxy raters

Extension focus:
- **Time decay** applied to fact scores before graph propagation

---

## Datasets

This repository contains **derived datasets only**, not the full original
HotpotQA training set.

Included:
- `data/hotpot_sample.json`  
  Small sample used for debugging and qualitative inspection.
- `data/d1_100.json`  
  100-sample clean dataset derived from HotpotQA-style inputs.
- `data/d2_20_contradictions.json`  
  20-sample dataset with synthetic contradictions for robustness analysis.

⚠️ **Note on original data**  
The full HotpotQA training file (`hotpot_train_v1.1.json`) is not included due to
its size and licensing constraints.  
To regenerate the datasets from scratch, place the original file in `data/`
and run:

```bash
python scripts/create_datasets.py