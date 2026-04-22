# Efficiency Audit Module

Evaluates whether an LLM prompt uses its context window efficiently. Counts tokens per section, detects redundant chunks, and measures how relevant retrieved context is to the query. Produces a JSON report consumed by the dashboard.



---

## Files

```
modules/efficiency/
│
├── src/
│   └── efficiency.py
├── data/
│   ├── nq/                       ← Natural Questions subset (Phase 1)
│   ├── longbench/                ← converted LongBench prompts (Phase 2)
│   └── synthetic/                ← three synthetic scenarios (unit tests)
├── notebooks/
│   └── efficiency_demo.ipynb
├── outputs/
│   └── reports/
├── tests/
│   └── test_efficiency.py
└── README.md
```

---

## Stack

Base data libraries (pandas, numpy, scikit-learn) are already in the root `requirements.txt`. Everything runs on CPU — no GPU, no external API calls.

The student picks one tokenizer and one embedding model.

**Token counting**

- `tiktoken` — OpenAI's tokenizer (GPT-3.5 / GPT-4 / GPT-4o). Fast, well-maintained, appropriate if the audited prompts target OpenAI models.
- Hugging Face `AutoTokenizer` — load the tokenizer of the specific model the team actually targets (Llama, Mistral, Claude via `transformers`, etc.). More accurate per-model, slightly more setup.

**Embedding model (via sentence-transformers)**

- `all-MiniLM-L6-v2` — smallest and fastest (~80 MB, 384-dim). Good baseline for a working prototype.
- `all-mpnet-base-v2` — larger (~420 MB, 768-dim) but higher semantic quality. Still CPU-feasible, just slower per batch.
- `bge-small-en-v1.5` — optimized specifically for retrieval tasks, which is what Check 3 measures. Size and speed similar to MiniLM-L6 with generally better retrieval scores on public benchmarks.

---

## Detection Methods

Three checks, run together. Each returns LOW / MEDIUM / HIGH. The pipeline aggregates them into a single `overall_risk`.

**Check 1 — Token Budget.** Counts tokens per section. Flags when one non-query section dominates the budget — often a sign of instruction bloat or over-retrieval.

**Check 2 — Redundancy Detection.** Embeds prompt chunks with a sentence-transformer model and flags near-duplicate pairs. Catches repeated instructions and duplicated retrieved passages.

**Check 3 — Context Relevance.** Embeds the query and each retrieved chunk, flags chunks that are not semantically related to the query. Karpukhin et al. (2020) established dense passage retrieval as the standard approach for RAG, but even strong retrievers return imperfect results — this check measures that gap.

---

## Risk Levels

- If any check returns HIGH → `overall_risk = HIGH` (significant inefficiency; refactor recommended).
- Otherwise, if any check returns MEDIUM → `overall_risk = MEDIUM` (moderate; review manually).
- Otherwise → `overall_risk = LOW` (no significant signals; does not mean the prompt is optimal).

---

## Output

Each pipeline run produces a JSON report written to `outputs/reports/`. The report contains the total token count, the per-section token breakdown, per-check results, the aggregated risk level, and an estimated token-savings number. The dashboard reads this file and renders it alongside fairness and leakage reports.

---

## Datasets

- **Natural Questions subset** (Kwiatkowski et al., 2019) — a small slice (10–20 examples) of real Google queries paired with Wikipedia passages. Directly formattable into RAG-style prompts with minimal preprocessing. Used for fast validation.
- **LongBench subset** — a public benchmark of long-context tasks, converted into RAG-style prompts (see Evaluation Strategy below). Used for robustness testing at realistic scale.
- **Synthetic scenarios** — three hand-built prompts with injected inefficiencies (bloated system, duplicated chunks, irrelevant context), used as unit tests.

---

## Evaluation Strategy

Two real-data phases plus synthetic unit tests. The three-tier structure mirrors standard ML evaluation practice: unit tests validate the detection logic, a small realistic dataset allows fast iteration, and a full long-context benchmark validates generalization.

**Phase 1 — Fast validation (Natural Questions).**
Natural Questions provides query–passage pairs that map cleanly onto RAG-style prompts. Run the pipeline on 10–20 examples to:
- confirm Check 1 (Token Budget) identifies bloat correctly
- confirm Check 2 (Redundancy) detects duplicated or overlapping chunks
- confirm Check 3 (Relevance) flags weakly related retrieved passages
- debug thresholds and stabilize outputs before scaling up

**Phase 2 — Long-context evaluation (LongBench, converted).**
LongBench is not originally retrieval-augmented, so each example is converted into a RAG-style prompt through a small pipeline:

1. Split the long document into fixed-size chunks (200–500 tokens — standard in production RAG practice).
2. Simulate retrieval to select the top-k chunks. Either BM25 (via `rank_bm25`, zero extra dependencies) or embedding similarity (reusing the sentence-transformer already loaded for Checks 2 and 3) works.
3. Assemble the prompt: system instruction + query + retrieved chunks.

Run the pipeline on this converted set to evaluate token distribution, redundancy across retrieved chunks, and semantic alignment between query and context at realistic long-context scale. Expected outcome: most prompts land at LOW or MEDIUM, with HIGH only when a real inefficiency pattern is present.

**Synthetic scenarios (unit tests).**
Three hand-built prompts with known injected inefficiencies:
- bloated system prompt → Check 1 must fire HIGH
- duplicated chunks → Check 2 must fire HIGH
- irrelevant context → Check 3 must fire HIGH

If a check does not fire on its synthetic scenario, the implementation is broken. Fix the logic before touching thresholds.

---

## References

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* Transactions of the Association for Computational Linguistics, 12, 157–173. https://aclanthology.org/2024.tacl-1.9/
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems, 33, 9459–9474. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
- Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6769–6781. https://aclanthology.org/2020.emnlp-main.550/
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992. https://aclanthology.org/D19-1410/
- Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., & Petrov, S. (2019). *Natural Questions: A Benchmark for Question Answering Research.* Transactions of the Association for Computational Linguistics, 7, 452–466. https://aclanthology.org/Q19-1026/

**Tools and datasets:**
- tiktoken: https://github.com/openai/tiktoken
- sentence-transformers: https://www.sbert.net/
- LongBench: https://github.com/THUDM/LongBench
- Natural Questions: https://ai.google.com/research/NaturalQuestions
