# DB‑SLM Database & Algorithms — Deep Technical Spec (v2)

> This document tightens the statistical core of DB‑SLM so it can stand as a production‑quality, tensor‑free pseudo‑LLM. It formalizes advanced smoothing/backoff, mixture components (cache/pointer, domain mixtures), quantization, and decoding. It also provides concrete SQL/ETL patterns so the code in `src/db_slm` can be implemented directly against SQLite (dev) and MariaDB (target).

---

## 0. Scope & Guarantees

**Goal.** Implement an autoregressive text generator using relational tables only, with statistical behavior approaching classic SLMs augmented by concept‑level scaffolding.

**No‑tensor constraint.** All probabilities are computed from counts and pre‑materialized statistics; no embeddings, no matrix multiplies.

**Latency strategy.** Per‑token queries are amortized with (a) top‑K materializations and (b) Level‑3 concept verbalization to emit multi‑token spans per query.

**Determinism.** Given fixed seeds and unchanged tables, decoding is reproducible.

---

## 1. Tokenization & Vocabulary

### 1.1 Recommended tokenizers

1. **Byte‑level (robust default)**

   * Tokens are single bytes (0–255) plus control tokens (`<BOS>`, `<EOS>`, `<PAD>`). No OOVs.
   * Pros: deterministic, locale‑agnostic, cheap to implement in SQL.
   * Cons: longer sequences; mitigated by concept verbalization.

2. **Word‑piece / BPE (optional advanced)**

   * Train BPE merges offline via SQL ETL (see 1.3) and store in `tbl_vocab` + `tbl_bpe_merges`.
   * Pros: shorter sequences, better language modeling.
   * Cons: ETL complexity; still tensor‑free.

### 1.2 Core tables

```sql
CREATE TABLE tbl_l1_vocabulary (
  token_id      INT PRIMARY KEY,
  token_text    TEXT UNIQUE NOT NULL,
  is_control    TINYINT DEFAULT 0,
  freq_global   BIGINT DEFAULT 0,
  -- optional: byte value 0-255 for byte-level tokenizer
  byte_value    TINYINT NULL
) ENGINE=InnoDB;

CREATE TABLE tbl_token_normalization (
  raw TEXT PRIMARY KEY,
  norm TEXT NOT NULL,
  reason TEXT  -- e.g., "lowercase", "unicode_nfkc", "trim"
) ENGINE=InnoDB;
```

### 1.3 (Optional) BPE merge training in SQL (sketch)

* Represent sequences as rows `(doc_id, position, token_id)`.
* Compute pair counts with a self‑join and GROUP BY.
* Select argmax pair; insert into `tbl_bpe_merges(rank, left_id, right_id, new_id)`.
* Re‑tokenize by rewriting pairs using a rolling window. Iterate for `M` merges.

This can be executed in batches; materialize `tbl_seq_tokens_current` each round to keep joins tractable.

---

## 2. Level‑1: N‑gram Statistics (Counts → Probabilities)

We standardize on **Modified Kneser–Ney (MKN)** as the primary smoother, with Jelinek–Mercer (JM) and Witten–Bell (WB) as fallbacks for sparse orders. All are derived from count tables.

### 2.1 Tables (per order `n`)

```sql
-- Canonical n-gram counts
CREATE TABLE tbl_l1_ng_counts_n (
  context_hash  BINARY(8) NOT NULL,
  next_token_id INT NOT NULL,
  count         BIGINT NOT NULL,
  updated_at    DATETIME,
  PRIMARY KEY (context_hash, next_token_id)
) ENGINE=Aria;

-- Unique continuation counts per token (for KN)
CREATE TABLE tbl_l1_continuations (
  token_id      INT PRIMARY KEY,
  num_contexts  BIGINT NOT NULL,
  last_rebuild  DATETIME
) ENGINE=Aria;

-- Counts-of-counts (Good–Turing stats) per order
CREATE TABLE tbl_l1_counts_of_counts (
  n_order   TINYINT NOT NULL,
  c_value   INT NOT NULL,       -- e.g., 1,2,3,>=4 bucketed
  num_ngrams BIGINT NOT NULL,
  PRIMARY KEY (n_order, c_value)
) ENGINE=Aria;

-- Discount parameters and backoff weights per order (post-ETL)
CREATE TABLE tbl_l1_mkn_params (
  n_order  TINYINT PRIMARY KEY,
  D1       DOUBLE,
  D2       DOUBLE,
  D3p      DOUBLE,
  total_contexts BIGINT,
  total_types    BIGINT,
  built_at DATETIME
) ENGINE=InnoDB;

-- Materialized probabilities (quantized) per order
CREATE TABLE tbl_l1_ng_probs_n (
  context_hash   BINARY(8) NOT NULL,
  next_token_id  INT NOT NULL,
  q_logprob      TINYINT NOT NULL,  -- quantized log10 prob (0-255)
  backoff_alpha  SMALLINT,          -- optional: quantized alpha for this context
  PRIMARY KEY (context_hash, next_token_id),
  KEY idx_ctx_topk (context_hash, q_logprob DESC)
) ENGINE=Aria;

-- Top-K head cache per context for O(1) retrieval
CREATE TABLE tbl_l1_ng_topk_n (
  context_hash   BINARY(8) NOT NULL,
  k_rank         TINYINT NOT NULL,      -- 1..K
  next_token_id  INT NOT NULL,
  q_logprob      TINYINT NOT NULL,
  PRIMARY KEY (context_hash, k_rank)
) ENGINE=Aria;
```

**Notes**

* Use separate `*_n` tables for each order n (2..N). `context_hash` is the 64‑bit hash over the last `n-1` tokens.
* Byte‑order your `BINARY(8)` consistently (big‑endian recommended) to enable range sharding.

### 2.2 Modified Kneser–Ney (MKN) formulas

For order `n` with context (h) and candidate token (w):

* **Discounts** (Heuristics from counts‑of‑counts):
  $$
  D_1 = 1 - \frac{2N_2}{N_1 + 2N_2},\quad
  D_2 = 2 - \frac{3N_3}{N_2 + 2N_3},\quad
  D_{3+} = 3 - \frac{4N_4}{N_3 + 2N_4}
  $$  
  where (N_c) is the number of n‑grams with count exactly (c) (or bucketed for (c\ge 3)).

* **Discounted probability**
  $$
  P_{KN}(w\mid h) = \frac{\max(c(h,w) - D(c), 0)}{c(h,*)} + \alpha(h) \cdot P_{KN}(w\mid h')
  $$
  with backoff context (h') dropping the oldest token.

* **Backoff weight**
  $$
  \alpha(h) = \frac{D_1 N_1(h,*) + D_2 N_2(h,*) + D_{3+} N_{3+}(h,*)}{c(h,*)}
  $$
  where (N_c(h,*)) counts the number of distinct followers of (h) with count (c).

* **Lower‑order base (continuation probability, order=1)**
  $$P_{cont}(w) = \frac{\#\{h: c(h,w) > 0\}}{\sum_{w'} \#\{h: c(h,w') > 0\}}$$  

### 2.3 SQL ETL to compute MKN parameters (sketch)

**Counts‑of‑counts**

```sql
INSERT INTO tbl_l1_counts_of_counts(n_order, c_value, num_ngrams)
SELECT :n AS n_order, c_bucket, COUNT(*)
FROM (
  SELECT CASE
           WHEN count=1 THEN 1
           WHEN count=2 THEN 2
           WHEN count=3 THEN 3
           ELSE 4
         END AS c_bucket
  FROM tbl_l1_ng_counts_n
) t
GROUP BY c_bucket;
```

**Continuation counts** (for unigrams base)

```sql
REPLACE INTO tbl_l1_continuations(token_id, num_contexts)
SELECT next_token_id, COUNT(DISTINCT context_hash)
FROM tbl_l1_ng_counts_2  -- bigram table
GROUP BY next_token_id;
```

**Backoff numerators (N_c(h,*))**

```sql
-- Per context, how many followers with count bucket 1/2/3+
CREATE TEMPORARY TABLE tmp_ctx_buckets AS
SELECT context_hash,
       SUM(count=1) AS n1,
       SUM(count=2) AS n2,
       SUM(CASE WHEN count>=3 THEN 1 ELSE 0 END) AS n3p,
       SUM(count) AS c_total
FROM tbl_l1_ng_counts_n
GROUP BY context_hash;
```

**Discounts (one‑time)**: compute (D_1,D_2,D_{3+}) from `tbl_l1_counts_of_counts` at order `n` and store in `tbl_l1_mkn_params`.

**Materialize probabilities (order n)**

```sql
-- First, compute the discounted relative term
CREATE TEMPORARY TABLE tmp_discounted AS
SELECT c.context_hash, c.next_token_id,
       GREATEST(c.count - CASE
          WHEN c.count=1 THEN p.D1
          WHEN c.count=2 THEN p.D2
          ELSE p.D3p END, 0) / b.c_total AS rel_prob,
       b.c_total, b.n1, b.n2, b.n3p, p.D1, p.D2, p.D3p
FROM tbl_l1_ng_counts_n c
JOIN tmp_ctx_buckets b USING (context_hash)
JOIN tbl_l1_mkn_params p ON p.n_order=:n;

-- Backoff alpha per context
CREATE TEMPORARY TABLE tmp_alpha AS
SELECT context_hash,
       (D1*n1 + D2*n2 + D3p*n3p) / NULLIF(c_total,0) AS alpha
FROM tmp_discounted
GROUP BY context_hash;

-- Lower‑order probability lookup is from tbl_l1_ng_probs_(n-1), or P_cont for unigrams
-- Here we assume we already have q_logprob for (n-1); we dequantize with a LUT.

-- Join to lower order and finalize P
CREATE TEMPORARY TABLE tmp_final AS
SELECT d.context_hash, d.next_token_id,
       (d.rel_prob + a.alpha * pow10_lookup(lo.q_logprob)) AS p_final
FROM tmp_discounted d
JOIN tmp_alpha a USING (context_hash)
JOIN tbl_l1_ng_probs_(n-1) lo
  ON lower_hash(d.context_hash) = lo.context_hash
 AND lo.next_token_id = d.next_token_id;

-- Quantize log10 probabilities into 0..255
REPLACE INTO tbl_l1_ng_probs_n(context_hash, next_token_id, q_logprob, backoff_alpha)
SELECT context_hash, next_token_id,
       quantize_log10(p_final) AS q_logprob,
       quantize_lin((SELECT alpha FROM tmp_alpha ta WHERE ta.context_hash=tmp_final.context_hash))
FROM tmp_final;
```

**Utilities**

* `lower_hash(BINARY(8))` drops the oldest token from the context (implemented by re‑hash or by storing token windows in a side table during ETL).
* `pow10_lookup()` and `quantize_log10()` are scalar UDFs or lookup tables mapping `TINYINT` to `DOUBLE` and back.

### 2.4 Interpolation alternatives

* **Jelinek–Mercer (JM)** per context: $P=\lambda P_n + (1-\lambda)P_{n-1}$ with global or context‑dependent $\lambda$. Store $\lambda$ in `tbl_l1_jm_params(n_order, lambda)`.
* **Witten–Bell (WB)** per context using the number of distinct followers: $(P=\frac{c}{c+T} + \frac{T}{c+T}P_{n-1})$, where (T) is distinct follower count. Requires only `n1+n2+n3+` from buckets.

### 2.5 Cache/Pointer mixture (session‑adaptive LM)

Implement a **pointer‑sentinel mixture** to bias towards recent tokens:

```sql
CREATE TABLE tbl_l1_session_cache (
  conversation_id  CHAR(36),
  token_id         INT,
  recency_weight   DOUBLE,
  PRIMARY KEY (conversation_id, token_id)
) ENGINE=InnoDB;
```

At decode step t for session s:
$$P_{final}(w) = \lambda_s P_{cache}(w \mid s) + (1-\lambda_s) P_{base}(w)$$

* `P_cache` from normalized `recency_weight` within a rolling window of last `W` tokens (exponential decay).
* $\lambda_s$ from a small heuristic table keyed by session length or entropy; store in `tbl_decode_hparams`.

### 2.6 Domain mixtures (mixture‑of‑LMs without tensors)

Partition the corpus by domain (e.g., code, wiki, chat) and train counts per domain. At inference, use a log‑linear mixture:
$\log P(w|h) = \sum_k \gamma_k(h) \log P_k(w|h) - A(h)$

Store `gamma_k` as normalized weights estimated from domain classifier rules (regex, URL, channel). Materialize top‑K per domain to keep joins small.

### 2.7 Repetition penalties & constraints

Maintain *soft bans* at decode time:

* **Presence penalty:** subtract a fixed `q_delta` if token already appeared in output.
* **Frequency penalty:** subtract `q_delta * count_in_output`.
* **Banned tokens:** a denylist table `tbl_decode_bans(profile, token_id)`.

Apply by adjusting `q_logprob` on the fly before sampling.

### 2.8 Top‑K and nucleus (top‑p) sampling without floats

* **Top‑K:** read from `tbl_l1_ng_topk_n`.
* **Top‑p:** cumulative mass over quantized probs using a LUT `tbl_q_to_mass(tinyint -> double)`. Stop when sum ≥ p.
* **Temperature T:** precompute LUTs `tbl_temp_lut(T, q_in -> q_out)` for a small grid of T (e.g., 0.7, 1.0, 1.3) using power transform before re‑normalization.

### 2.9 Optional: Alias method for O(1) sampling on hot contexts

For the top 1e6 hottest contexts, build `tbl_alias_n` with per‑bucket `(prob, alias_token_id)` rows. Use during decode when cache hits; fall back to Top‑K otherwise.

---

## 3. Level‑2: Memory, Corrections, and Logit‑Bias via SQL

**Key addition:** convert user corrections and profiles into **token‑ or span‑level logit biases** applied during decoding.

```sql
CREATE TABLE tbl_l2_token_bias (
  conversation_id CHAR(36),
  pattern         TEXT,        -- LIKE/REGEXP matching decoded context
  token_id        INT,
  q_bias          SMALLINT,    -- signed quantized delta in log10 space
  expires_at      DATETIME,
  PRIMARY KEY (conversation_id, pattern, token_id)
) ENGINE=InnoDB;
```

* When a correction is logged, materialize bias entries to up‑weight facts (e.g., prefer token_id for “Paris”).
* At decode, join the current context string against `pattern` and adjust `q_logprob`.

**Rolling prompt window** remains defined in `tbl_l2_messages` with MyRocks; keep a denormalized `tbl_l2_window_cache(conversation_id, window_text)` to reduce per‑turn JOINs.

---

## 4. Level‑3: Concept Model as Class‑Based LM

Treat each concept as a **class token** and train a concept n‑gram over the same hashed contexts.

* `tbl_l3_concept_repo`, `tbl_l3_verbal_templates` as catalog + render.
* `tbl_l3_concept_probs_n(context_hash, concept_id, q_logprob)` analogous to word LM.
* **Execution signals** are high‑priority overrides (already specified) to guarantee actions.

**Training:** capture executed concepts per turn, hash their preceding tokens (or preceding concepts), and upsert counts into `tbl_l3_concept_counts_n`, then run the same MKN ETL.

**Verbalization:** use templates for multi‑token spans; append a small connector sequence using Level‑1 stitching.

---

## 5. End‑to‑End Decoding

### 5.1 Pseudocode

```
Respond(session s, user_text):
  log user_text into tbl_l2_messages
  context_tokens ← tokenize(window(s))

  # 1) Concept step (optional each segment)
  concept ← argmax_concept(context_tokens) or consume_signal(s)
  if concept:
      payload ← build_payload(concept, s)
      span ← verbalize(concept, payload)
      emit(span)
      extend(context_tokens, tokenize(span))

  # 2) Stitching with Level‑1
  while not stop:
      h ← last (n-1) tokens
      topk ← fetch_topk(h, n)
      q ← apply_temperature(topk.q_logprob, T)
      q ← apply_biases(q, s)
      q ← apply_penalties(q, output)
      next ← sample(q, method=top_p)
      emit(next)
      extend(context_tokens, next)
      if next == <EOS> or length ≥ max_len: break

  log assistant output
```

### 5.2 SQL for Top‑K fetch

```sql
SELECT next_token_id, q_logprob
FROM tbl_l1_ng_topk_n
WHERE context_hash = :h
ORDER BY k_rank ASC
LIMIT :K;
```

If miss, fallback:

```sql
SELECT next_token_id, q_logprob
FROM tbl_l1_ng_probs_n
WHERE context_hash = :h
ORDER BY q_logprob DESC
LIMIT :K;
```

---

## 6. Training/ETL Pipeline

1. **Ingest & normalize** → `tbl_corpus(doc_id, text)`; populate `tbl_token_normalization`.
2. **Tokenize** → `tbl_seq_tokens(doc_id, pos, token_id)` with deterministic tokenizer.
3. **Extract n‑grams** for each order n: streaming window into `tbl_l1_ng_counts_n` via UPSERT/batch.
4. **Continuation counts** from bigrams → `tbl_l1_continuations`.
5. **Counts‑of‑counts** per n → `tbl_l1_counts_of_counts`.
6. **Compute MKN params** per n → `tbl_l1_mkn_params`.
7. **Materialize probs** per n → `tbl_l1_ng_probs_n` and **Top‑K** → `tbl_l1_ng_topk_n`.
8. **(Hot) Alias tables** for frequent contexts (optional).
9. **Pruning jobs** (see next) to control size.

### 6.1 Pruning strategies

* **Entropy pruning:** drop n‑grams whose removal increases cross‑entropy < ε.
* **Absolute threshold:** keep counts ≥ k.
* **Stupid backoff guard:** for ultra‑rare contexts, skip materialization and rely on lower orders.

### 6.2 Incremental updates

* Append new corpus batches into `tbl_l1_ng_counts_n_delta`; nightly merge into base with `INSERT…ON DUPLICATE KEY UPDATE count = count + VALUES(count)`.
* Rebuild Top‑K for contexts whose totals changed above a threshold.

### 6.3 Emotion‑rich bootstrap corpus (`datasets/emotion_data.json`)

* **Format.** Newline‑delimited JSON where each row contains `prompt`, `emotion`, and `response`. The `response` text already carries long‑form reasoning tied to the emotion label, making it ideal for kick‑starting Level‑1 statistics and Level‑3 concept cues.
* **Normalization pass.**
  1. Read JSON rows via Python (stream + `json.loads`) so we can record ingestion provenance.
  2. Store `prompt || '\n\n' || response` into `tbl_corpus.text`, while persisting `emotion` inside `tbl_corpus_meta(doc_id, key, value)` to keep the label queryable.
  3. Optionally insert the `prompt` alone into `tbl_l2_messages` as a seed “user” turn so inference smoke tests can replay it verbatim.
* **Training split.** Reserve ~5 % of the rows (systematic sampling by doc_id hash) as a held‑out inference suite. The remaining 95 % flow through the standard pipeline: tokenize → extract n‑grams → recompute MKN parameters.
* **Inference harness.**
  * During decoding, replay a held‑out `prompt` and check whether the generated continuation overlaps qualitatively with the reference `response`. Track success via ROUGE‑L / overlap of sentiment adjectives.
  * Record per‑emotion coverage: `SELECT emotion, COUNT(*)` from the held‑out table joined against decoded runs. This ensures jealousy/apathy/etc. all exercise the pointer/cache logic before we move to larger corpora.
* **Automation hook.** `src/db_slm/settings.py` exposes `DBSLM_DATASET_PATH` so CLI tools (or dedicated ETL scripts) always pick up the same JSON file defined in `.env`. This keeps experimentation reproducible when we swap in more datasets later.

---

## 7. Quantization & Lookup LUTs

### 7.1 Log‑prob quantization

Map real log10 probabilities in [−20, 0] → `TINYINT [0,255]`:

* `q = round( clamp((log10(p) - Lmin) / (Lmax - Lmin), 0, 1) * 255 )`
* Store `Lmin,Lmax` in `tbl_quant_meta(name, Lmin, Lmax)`.

### 7.2 LUT tables

```sql
CREATE TABLE tbl_q_to_mass (
  q TINYINT PRIMARY KEY,
  prob DOUBLE NOT NULL,
  log10 DOUBLE NOT NULL
);

CREATE TABLE tbl_temp_lut (
  temp DECIMAL(3,1),
  q_in  TINYINT,
  q_out TINYINT,
  PRIMARY KEY (temp, q_in)
);
```

---

## 8. Performance & Scaling

* **Engines:** Aria for `*_counts_*`, `*_probs_*`, `*_topk_*`; MyRocks for `tbl_l2_messages`; InnoDB for control/meta.
* **Partitioning:** HASH on `context_hash` into P shards; keep Top‑K co‑located.
* **Keys:** `(context_hash, q_logprob DESC)` covering index for fast head scans; `PRIMARY KEY(context_hash, next_token_id)` for UPSERTs.
* **I/O:** Use compressed row formats; pin top shard pages in page cache.
* **Concurrency:** WAL for SQLite dev; MariaDB with large `aria_pagecache_buffer_size`.
* **Batching:** Favor bulk INSERTs; defer probability builds to offline windows.

---

## 9. Evaluation

* **Perplexity** on held‑out set:

```sql
SELECT EXP( - AVG( ln_prob ) ) AS ppl
FROM (
  SELECT SUM( ln(pow10_lookup(q_logprob)) ) AS ln_prob
  FROM eval_next_token_labels e
  JOIN tbl_l1_ng_probs_n p
    ON p.context_hash=e.context_hash AND p.next_token_id=e.gold_token
  GROUP BY e.sequence_id
) s;
```

* **Coverage:** % gold tokens missing at order n (signals backoff issues).
* **Latency:** p50/p95 per decode step from query logs.

---

## 10. API Surfaces (for `src/db_slm`)

* `NGramStore`: get_topk(h, n), get_prob(h, w, n), rebuild_topk(context_hashes[])
* `Smoother`: mkn_materialize(n), jm_materialize(n, λ), wb_materialize(n)
* `CacheLM`: update_session(s, token_id), mixture_weights(s)
* `Bias`: upsert_bias(s, pattern, token_id, q_bias)
* `Decoder`: sample_topk(), sample_topp(), apply_temperature(), apply_penalties()
* `ConceptModel`: predict_concept(h), render(concept_id, payload)

---

## 11. Migration to MariaDB (from SQLite dev)

* Replace `ON CONFLICT` with `INSERT … ON DUPLICATE KEY UPDATE`.
* Convert BLOB hashes to `BINARY(8)`.
* Move temp tables to `MEMORY` engine where safe.
* Create stored functions for LUTs; otherwise use small join tables.

---

## 12. Roadmap

1. Implement MKN ETL (2.3) and Top‑K materialization end‑to‑end.
2. Add session cache & pointer mixture (2.5) and bias application (3).
3. Train concept n‑gram and wire to templates (4).
4. Introduce entropy pruning and nightly rebuild jobs.
5. Optional: alias method for top 1e6 contexts; BPE tokenizer.

---

**Deliverable status:** This spec is sufficient to (a) fill all probability tables from counts, (b) decode with MKN + top‑K/top‑p, (c) bias outputs via memory, and (d) amortize latency with concept spans. It keeps the system tensor‑free while bringing it closer to the behavior of modern text generators.

## 13. Reference Pseudocode (Engine, Training, Inference)

The following pseudocode threads the CONCEPT blueprint, this spec, and the upcoming `src/db_slm` implementation. Each block is an executable recipe for the core algorithms: booting the database engine, running the statistical training/ETL loop, and serving inference with Level‑3 concepts plus Level‑2 memory.

### 13.1 Engine Boot & Maintenance

```text
procedure InitializeEngine(cfg):
    conn ← open_database(cfg.dsn)
    apply_sqlite_pragmas(conn, wal=ON, synchronous=NORMAL)
    run_migrations(conn, MIGRATIONS_DIR)

    vocab ← load_vocabulary(conn)
    quant_luts ← load_quantization_tables(conn)
    concept_repo ← load_concept_catalog(conn)

    hot_contexts ← fetch_hot_contexts(conn, cfg.hot_rank_threshold)
    prediction_cache ← hydrate_topk_cache(conn, hot_contexts)

    bias_index ← preload_active_biases(conn, since=cfg.bias_window)
    return Engine(conn, vocab, quant_luts, concept_repo,
                  prediction_cache, bias_index, cfg)

procedure MaintenanceTick(engine, now):
    dirty_orders ← collect_dirty_orders(engine.changelog, now)
    if dirty_orders ≠ ∅:
        lock(engine.maintenance_mutex)
        for n in dirty_orders:
            dirty_ctx ← pop_dirty_contexts(engine.changelog, n)
            RebuildProbabilities(engine, n, dirty_ctx)
            RefreshTopK(engine, n, dirty_ctx)
        unlock(engine.maintenance_mutex)

    if now - engine.last_prune ≥ engine.cfg.prune_interval:
        run_entropy_pruning(engine.conn, engine.cfg.entropy_epsilon)
        engine.last_prune ← now

    decay_bias_cache(engine.bias_index, now)
```

*`dirty_orders`* is fed by training jobs inserting into `tbl_l1_ng_counts_n_delta`. `RebuildProbabilities` and `RefreshTopK` reuse the ETL steps in §2.3 and §6, while the cache hydration mirrors the adaptive cache already described in `AI_REFERENCE.md`.

### 13.2 Training & ETL Loop

```text
procedure TrainBatch(engine, corpus_batch):
    normalized_batch ← [normalize_text(doc) for doc in corpus_batch]
    token_stream ← tokenize_batch(normalized_batch, engine.vocab)

    context_registry_updates ← {}
    for n in 1..engine.cfg.max_order:
        ngram_counts ← slide_window(token_stream, n)
        upsert_counts(engine.conn, table=tbl_l1_ng_counts_n, rows=ngram_counts)
        mark_dirty(engine.changelog, order=n, contexts=ngram_counts.context_hashes)
        accumulate_registry(context_registry_updates, ngram_counts)

    upsert_context_registry(engine.conn, context_registry_updates)
    log_training_batch(engine.conn, corpus_batch.meta)

procedure RebuildProbabilities(engine, order_n, target_contexts):
    compute_counts_of_counts(engine.conn, order_n, scope=target_contexts)
    discounts ← solve_mkn_discounts(engine.conn, order_n)
    ctx_buckets ← materialize_ctx_buckets(engine.conn, order_n, target_contexts)

    discounted ← compute_discounted_mass(engine.conn, order_n, ctx_buckets, discounts)
    alpha ← compute_backoff_alpha(discounted, ctx_buckets, discounts)
    lower_order ← fetch_lower_order_probs(engine.conn, order_n-1, discounted.tokens)

    final_mass ← apply_backoff(discounted, alpha, lower_order)
    quantized ← quantize_probs(final_mass, engine.quant_luts)
    replace_into_probs(engine.conn, order_n, quantized)

procedure RefreshTopK(engine, order_n, target_contexts):
    for ctx in target_contexts:
        topk ← select_topk(engine.conn, order_n, ctx, K=engine.cfg.default_k)
        replace_into_topk(engine.conn, order_n, ctx, topk)
        if ctx in engine.prediction_cache:
            engine.prediction_cache[ctx] ← topk
```

This loop mirrors §6: ingest text, extract n‑grams, update auxiliary stats, then rebuild `tbl_l1_ng_probs_n` and `tbl_l1_ng_topk_n`. The helper routines map directly to SQL fragments already described in §2.3 (counts‑of‑counts, discounted mass, alpha, quantization).

### 13.3 Inference, Concept Execution, and Decoding

```text
procedure RunSession(engine, conversation_id, user_text):
    log_message(engine.conn, conversation_id, role="user", text=user_text)
    context_tokens ← window_tokens(engine.conn, conversation_id,
                                   engine.cfg.window_size)

    while session_active(conversation_id):
        concept ← MaybeSelectConcept(engine, context_tokens, conversation_id)
        if concept ≠ NONE:
            span ← ExecuteConcept(engine, concept, conversation_id)
            emit_to_client(span)
            context_tokens ← append_tokens(context_tokens, tokenize(span))
            continue

        word ← DecodeLevel1(engine, conversation_id, context_tokens)
        emit_to_client(engine.vocab[word])
        context_tokens ← append_tokens(context_tokens, [word])

        if stop_condition(word, context_tokens, engine.cfg):
            break

    log_message(engine.conn, conversation_id, role="assistant",
                text=render_output(conversation_id))

procedure MaybeSelectConcept(engine, ctx_tokens, conversation_id):
    signals ← poll_execution_signals(engine.conn, conversation_id)
    if signals ≠ ∅:
        return highest_priority(signals)

    concept_topk ← fetch_concept_topk(engine.conn, ctx_tokens)
    scored ← apply_concept_wpriors(concept_topk, conversation_id)
    if confidence(scored.best) ≥ engine.cfg.concept_threshold:
        return scored.best
    return NONE

procedure ExecuteConcept(engine, concept, conversation_id):
    payload ← hydrate_concept_payload(engine.conn, concept, conversation_id)
    span ← verbalize_concept(engine.concept_repo, concept, payload)
    store_concept_trace(engine.conn, conversation_id, concept, payload, span)
    return span

procedure DecodeLevel1(engine, conversation_id, ctx_tokens):
    h ← hash_context(ctx_tokens, order=engine.cfg.max_order)
    topk ← engine.prediction_cache.get(h)
    if topk = NONE:
        topk ← fetch_topk(engine.conn, h, engine.cfg.max_order)
        engine.prediction_cache[h] ← topk

    q ← apply_temperature(topk.q_logprob, engine.cfg.temperature)
    q ← apply_biases(q, conversation_id, engine.bias_index)
    q ← apply_pointer_mixture(q, ctx_tokens, engine.cfg.cache_lambda)
    q ← apply_penalties(q, ctx_tokens, engine.cfg.penalties)

    choice ← sample_top_p(q, engine.cfg.top_p, engine.quant_luts)
    update_session_cache(engine.conn, conversation_id, choice)
    return choice
```

`RunSession` extends the earlier §5.1 loop with concrete hooks: Level‑3 concept arbitration (`MaybeSelectConcept`), Level‑2 corrections via `apply_biases`, pointer/cache mixture (§2.5), and quantized sampling. The same routines back the CLI/REPL flow mentioned in `AI_REFERENCE.md`, ensuring the documentation, concept study, and source all point to identical algorithms.

---

## 14. Environment‑driven configuration

* Copy `.env.example` → `.env` and fill the knobs: the active backend (`DBSLM_BACKEND`), `DBSLM_SQLITE_PATH`, the canonical emotional dataset (`DBSLM_DATASET_PATH`), and the cheetah transport fields (`DBSLM_CHEETAH_*`).
* `src/db_slm/settings.py::load_settings` is a zero‑dependency parser that reads `.env`, overlays real process env variables, and surfaces a `DBSLMSettings` dataclass. `train.py` / `run.py` already draw their default DB paths from `sqlite_dsn()`, so switching datasets or DB paths is a file edit rather than CLI surgery.
* Keep `.env` out of version control; `.env.example` documents every knob while leaving secrets blank. This allows CI to inject passwords via env vars and local devs to pin paths that live on their workstation.
