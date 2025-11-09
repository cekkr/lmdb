# Database Layout and Algorithmic Playbook for DB-SLM

This note extends the architectural blueprint in `studies/CONCEPT.md` with a concrete storage
layout and the end-to-end algorithms already prototyped under `src/db_slm`. The goal is to close
the remaining design gap for `src/db_slm` by documenting how the relational schema, the training ETL,
and the inference loop emulate the statistical behaviors normally delivered by GPT-class models while
remaining tensor-free.

---

## 1. Guiding Assumptions

- **Autoregressive objective:** As with GPT, generation is a next-token prediction problem over a
  finite vocabulary. Probabilities are expressed through conditional counts instead of softmaxed
  logits.
- **Three-level split:** Level 1 handles statistical continuity (Aria), Level 2 captures episodic and
  semantic memory (MyRocks + InnoDB), and Level 3 performs concept selection plus template
  verbalization (InnoDB). This mirrors the hierarchy laid out in `studies/CONCEPT.md`.
- **Engine hints, not engines:** SQLite stands in for MariaDB, so every table carries an
  `engine_hint` column to preserve the intent (read-heavy vs write-heavy vs transactional) for later
  migration.
- **GPT knowledge base:** Concepts such as prefill vs decode, KV caching, and sampling temperature
  inform our schema choices (hashed context registries, cache invalidation, conversation-scoped
  signals).

---

## 2. Database Structure

### 2.1 Engine Simulation and Pragmas

- WAL mode + `synchronous=NORMAL` recreate Aria-style concurrent reads (`src/db_slm/db.py`).
- `engine_hint` columns mark intended backends: Aria for Level 1/3 probability slabs, MyRocks for
  append-heavy chat logs, InnoDB for strongly consistent metadata and corrections.
- Context hashes use SHA-1 over normalized tokens; this is the relational analogue to GPT's KV cache
  key.

### 2.2 Level 1 — Statistical N-grams (Aria)

| Table | Purpose | Notable Columns | Indexing |
| --- | --- | --- | --- |
| `tbl_l1_context_registry` | Hot context directory storing `(context_hash, order_size, total_count, hot_rank)` so denominators for Laplace smoothing are cached. | `hot_rank` is recomputed from recent accesses to mimic GPT cache eviction. | `idx_l1_context_heat` orders by `(hot_rank DESC, last_seen_at DESC)` for prefetching. |
| `tbl_l1_ngram_counts` | Canonical `(context_hash, next_token)` counts, updated via UPSERT during training. | `observed_count` (REAL for now, quantized later), `last_seen_at`. | `idx_l1_counts_context` keeps the `(context_hash, observed_count DESC)` scan tight. |
| `tbl_l1_ngram_probs` | Optional materialization of smoothed probabilities, ready for bulk export or analytical tuning. | `probability` (REAL, future `TINYINT`). | `idx_l1_context` filters by hash. |

Design notes:

1. The registry replaces GPT's denominator recomputation by persisting `total_count`.
2. Counts are authoritative; n-gram probabilities are computed on read using
   `denom = total_count + alpha * variant_count`.
3. Prediction caching in `NGramModel` is keyed by `(context_hash, limit)` and invalidated whenever the
   same context reappears during training—similar to clearing GPT KV caches when the prompt changes.

### 2.3 Level 2 — Stateful Memory (MyRocks + InnoDB)

| Table | Engine intent | Description |
| --- | --- | --- |
| `tbl_l2_conversations` | InnoDB | Conversation metadata (`id`, `user_id`, `agent_name`, timestamps). |
| `tbl_l2_messages` | MyRocks | Append-only log of user/assistant turns; order by `(conversation_id, created_at)` for fast windows. |
| `tbl_l2_correction_log` | InnoDB | Stores explicit corrections (`corrected_fact_json`) so the model can override Level 1 tokens. |

Key behaviors:

- Conversation windows (`ConversationMemory.context_window`) implement GPT's rolling prompt
  truncation via SQL, keeping the last `N` turns.
- `lookup_corrections` feeds the concept payload providers before token generation, mirroring how
  GPT-based RAG systems inject retrieved facts ahead of decoding.

### 2.4 Level 3 — Conceptual Generation (InnoDB + Aria)

| Table | Use |
| --- | --- |
| `tbl_l3_concept_repo` | Catalog of semantic concepts (e.g., `ContextSummary`, `CorrectionReplay`) with JSON schemas. |
| `tbl_l3_verbal_templates` | Language-specific templates that verbalize concept payloads. |
| `tbl_l3_concept_probs` | Aria-style concept `n`-gram probabilities keyed by `context_hash`. |
| `tbl_l3_concept_signals` | High-priority hints pushed by Level 2 events; enforce concept selection before statistics. |

The Concept Engine first consumes signals (short-lived overrides), then falls back to statistical
concept prediction (`ConceptPredictor.predict`). Once a concept is chosen, the registered payload
provider gathers the necessary fields—often by querying Level 2—and renders the template. This is the
hierarchical \"concept → verbalization → token stitching\" loop defined in the Concept study.

### 2.5 Derived Structures and Caches

- **Prediction cache:** Prevents repeated SQL for hot contexts, analogous to GPT KV reuse.
- **Default context rows:** `context_hash="__default__"` seeds fallback predictions for cold starts.
- **Smoothing parameters:** `smoothing_alpha` in `NGramModel` implements additive smoothing to avoid
  zero-probability contexts, mirroring GPT's logit bias floor.

### 2.6 Table Specification Reference

#### Level 1 (Aria hint)

`tbl_l1_context_registry`
- Columns: `context_hash TEXT PK`, `order_size INTEGER`, `total_count INTEGER`, `last_seen_at TEXT`,
  `hot_rank REAL`, `engine_hint TEXT DEFAULT 'Aria'`.
- Indexes: `idx_l1_context_heat(hot_rank DESC, last_seen_at DESC)`.
- Notes: Holds one row per `(n-1)`-token window and acts as the denominator cache for smoothing.

`tbl_l1_ngram_counts`
- Columns: `context_hash TEXT`, `next_token TEXT`, `observed_count REAL`, `last_seen_at TEXT`,
  `engine_hint TEXT DEFAULT 'Aria'`.
- Constraints: `PRIMARY KEY(context_hash, next_token)`.
- Indexes: `idx_l1_counts_context(context_hash, observed_count DESC)`.
- Notes: Primary training target; each upsert increments `observed_count`.

`tbl_l1_ngram_probs`
- Columns: `id INTEGER PK AUTOINCREMENT`, `context_hash TEXT`, `next_token TEXT`, `probability REAL`,
  `engine_hint TEXT DEFAULT 'Aria'`.
- Indexes: `idx_l1_context(context_hash)`.
- Notes: Optional materialization layer for offline analytics or future quantization.

#### Level 2 (MyRocks + InnoDB hints)

`tbl_l2_conversations`
- Columns: `id TEXT PK`, `user_id TEXT`, `agent_name TEXT`, `created_at TEXT DEFAULT CURRENT_TIMESTAMP`,
  `engine_hint TEXT DEFAULT 'InnoDB'`.
- Notes: One row per session; referenced by all Level 2/3 child tables.

`tbl_l2_messages`
- Columns: `id TEXT PK`, `conversation_id TEXT`, `sender TEXT CHECK sender IN ('user','assistant')`,
  `content TEXT`, `created_at TEXT`, `engine_hint TEXT DEFAULT 'MyRocks'`.
- Indexes: `idx_l2_messages_conv(conversation_id, created_at)`.
- Notes: Represents the append-only turn log; chronological scans reconstruct the prompt window.

`tbl_l2_correction_log`
- Columns: `correction_id TEXT PK`, `conversation_id TEXT`, `error_message_id TEXT`,
  `correction_message_id TEXT`, `error_context TEXT`, `corrected_fact_json TEXT`,
  `created_at TEXT`, `engine_hint TEXT DEFAULT 'InnoDB'`.
- Notes: Stores structured corrections that override statistical predictions.

#### Level 3 (InnoDB + Aria hints)

`tbl_l3_concept_repo`
- Columns: `concept_id INTEGER PK AUTOINCREMENT`, `concept_name TEXT UNIQUE`,
  `metadata_schema TEXT`, `engine_hint TEXT DEFAULT 'InnoDB'`.
- Notes: Catalog of semantic actions; schemas describe required payload fields.

`tbl_l3_verbal_templates`
- Columns: `template_id INTEGER PK AUTOINCREMENT`, `concept_id INTEGER`, `template_string TEXT`,
  `language_code TEXT DEFAULT 'en'`, `engine_hint TEXT DEFAULT 'InnoDB'`.
- Indexes: `idx_l3_templates(concept_id, language_code)`.
- Notes: Stores language-specific render strings; `{placeholders}` map to payload keys.

`tbl_l3_concept_probs`
- Columns: `id INTEGER PK AUTOINCREMENT`, `context_hash TEXT`, `next_concept_id INTEGER`,
  `quantized_prob REAL`, `engine_hint TEXT DEFAULT 'Aria'`.
- Indexes: `idx_l3_context(context_hash)`.
- Notes: Mirrors Level 1 but over concept IDs; future work quantizes `quantized_prob` to bytes.

`tbl_l3_concept_signals`
- Columns: `signal_id TEXT PK`, `conversation_id TEXT`, `concept_id INTEGER`, `score REAL`,
  `expires_at TEXT`, `consume_once INTEGER`, `created_at TEXT`, `engine_hint TEXT DEFAULT 'InnoDB'`.
- Indexes: `idx_l3_signals_conv(conversation_id, score DESC)`.
- Notes: Implements deterministic overrides so that corrections or workflows can force a concept.

---

## 3. Training Algorithm

### 3.1 Corpus ETL (Level 1)

`src/train.py` drives ingestion through the following pipeline:

1. **Tokenization:** Lowercase regex tokenization (`\w+|[^\w\s]`) ensures punctuation is preserved,
   mirroring GPT's byte-pair boundary sensitivity.
2. **Sliding window:** For each order-`n` window, split into `(context_tokens, next_token)`.
3. **Hashing:** Compute `context_hash = SHA1(context_tokens)` to keep lookups O(1) regardless of
   context length.
4. **Upserts:**
   - Increment `tbl_l1_context_registry.total_count` by the window weight.
   - Increment `tbl_l1_ngram_counts.observed_count` for the `(context_hash, next_token)` pair.
5. **Cache invalidation:** Remove cached predictions for the mutated context hash.

Compared with GPT pre-training:

- We substitute explicit count increments for gradient updates.
- The \"loss\" is implicit: high-frequency contexts dominate, while low-frequency contexts are pruned
  by not ingesting them (or by future scheduled pruning jobs).

### 3.2 Probability Materialization (Optional)

Although the runtime calculator derives probabilities from counts, large deployments can materialize
`tbl_l1_ngram_probs` via:

```sql
INSERT INTO tbl_l1_ngram_probs(context_hash, next_token, probability)
SELECT
    c.context_hash,
    c.next_token,
    (c.observed_count + :alpha) /
    (r.total_count + :alpha * variant_count) AS probability
FROM tbl_l1_ngram_counts AS c
JOIN tbl_l1_context_registry AS r USING (context_hash);
```

This mirrors GPT's practice of periodically writing checkpointed weights for faster inference.

### 3.3 Concept Model Training

- Seeds register baseline concepts (`ContextSummary`, `CorrectionReplay`) with templates and default
  probabilities tied to `__default__` contexts (see `DBSLMEngine._ensure_concept_defaults`).
- Future training can log concept executions, compute their context hashes, and persist deltas in
  `tbl_l3_concept_probs`, effectively building an \"n-gram of concepts\".

### 3.4 Correction Logging Workflow

1. User issues a correction; Level 2 stores it via `ConversationMemory.record_correction`.
2. `DBSLMEngine.record_correction` pushes a `tbl_l3_concept_signals` row so the upcoming inference run
   deterministically selects `CorrectionReplay`.
3. Corrections remain queryable for payload generation, acting as a structured memory bank and
   eliminating the need for GPT-style gradient updates.

### 3.5 Pseudocode Reference

```pseudo
procedure TrainCorpus(corpus_text, order, db)
    tokens ← Tokenize(corpus_text)
    if length(tokens) < order:
        return
    for window in SlidingWindows(tokens, order):
        context_tokens ← window[0 : order-1]
        next_token ← window[-1]
        hash ← HashTokens(context_tokens)
        Upsert(tbl_l1_context_registry, hash, order-1, increment=1)
        Upsert(tbl_l1_ngram_counts, (hash, next_token), increment=1)
        InvalidatePredictionCache(hash)

procedure Upsert(table, key, order_size=None, increment)
    if table == tbl_l1_context_registry:
        execute SQL:
            INSERT ... ON CONFLICT(context_hash)
            DO UPDATE SET total_count = total_count + increment,
                         last_seen_at = CURRENT_TIMESTAMP
    else if table == tbl_l1_ngram_counts:
        execute SQL:
            INSERT ... ON CONFLICT(context_hash, next_token)
            DO UPDATE SET observed_count = observed_count + increment,
                         last_seen_at = CURRENT_TIMESTAMP

procedure RegisterConcept(concept_name, schema, template, default_prob)
    concept_id ← repo.register(concept_name, schema)
    verbalizer.register_template(concept_id, template)
    predictor.record_probability("__default__", concept_id, default_prob)
```

---

## 4. Inference Algorithm

### 4.1 Request Flow (Mirrors GPT Prefill → Decode)

1. **Log prompt:** Insert the user turn into `tbl_l2_messages`.
2. **Context window:** Fetch the latest `N` messages; tokenize to obtain the \"prefill\" context.
3. **Concept stage (hierarchical top-level):**
   - Consume pending signals (forced concepts).
   - Else, compute `context_hash` and query `tbl_l3_concept_probs` for the most probable concept.
   - Render the concept text via templates and payload providers.
4. **Token stitching (low-level decode):**
   - Append concept text tokens to the context.
   - Repeatedly query `NGramModel.predict_next` to extend the response with connector tokens (limit
     set via `target_length`), analogous to GPT's iterative decoding with temperature-free sampling.
5. **Finalize:** Join concept text + stitches, log the assistant message, update caches.

### 4.2 Sampling and Determinism

- Concept selection is greedy (highest probability), but signals allow deterministic overrides.
- Token stitching uses weighted random sampling proportional to smoothed probabilities. This behaves
  like GPT's nucleus sampling with a low temperature—diversity is limited by the top `limit`
  parameter passed to `predict_next`.

### 4.3 Incorporating Corrections

- Before payload rendering, `ConversationMemory.correction_digest` injects structured corrections.
- Payload providers can turn those JSON blobs into template fields, ensuring the assistant references
  the latest user feedback without waiting for Level 1 retraining.

### 4.4 Failure and Cold-Start Handling

- If no concept probability rows exist for a context, the engine falls back to `__default__`.
- If the Level 1 predictor cannot find rows for the stitched context, `seed_defaults` ensures
  generic discourse markers keep the reply grammatical.

### 4.5 Pseudocode Reference

```pseudo
function Respond(conversation_id, user_message):
    message_id ← memory.log_message(conversation_id, "user", user_message)
    context_text ← memory.context_window(conversation_id)
    context_tokens ← Tokenize(context_text)
    concept_exec ← RunConceptLayer(conversation_id, context_tokens)
    concept_text ← ""
    if concept_exec exists:
        concept_text ← concept_exec.text
        context_tokens ← context_tokens + Tokenize(concept_exec.text)
    stitching ← level1.stitch_tokens(context_tokens, target_length=18)
    response ← JoinSegments(concept_text, stitching)
    memory.log_message(conversation_id, "assistant", response)
    return response

function RunConceptLayer(conversation_id, context_tokens):
    hash ← HashTokens(context_tokens)
    signal ← pop_next_signal(conversation_id)
    if signal exists:
        concept ← repo.fetch_by_id(signal.concept_id)
        probability ← signal.score
    else:
        prediction ← predictor.predict(hash, fallback="__default__")
        if prediction is None:
            return None
        concept ← repo.fetch_by_id(prediction.concept_id)
        probability ← prediction.probability
    payload ← payload_provider(conversation_id, concept, context_tokens, memory)
    text ← verbalizer.render(concept.concept_id, payload)
    return {name: concept.name, text: text, probability: probability}

function StitchTokens(context_tokens, target_length):
    output ← []
    current ← context_tokens[-(order-1):]
    repeat target_length times:
        preds ← PredictNext(current, limit=3)
        if preds empty: break
        token ← WeightedSample(preds)
        append token to output
        append token to current; current ← last (order-1) tokens
    return Detokenize(output)
```

---

## 5. Implementation Status and Next Steps

1. **Schema completeness:** All Level 1–3 tables from `studies/CONCEPT.md` now exist with simulated
   engine hints. The remaining delta is materializing quantized probabilities (`TINYINT`) once we
   migrate off SQLite.
2. **Algorithm parity:** Training and inference loops replicate GPT behaviors (autoregressive next
   token, greedy concept selection, smoothing), albeit using relational queries instead of tensor
   kernels.
3. **Research tasks:**
   - Record concept execution traces to train `tbl_l3_concept_probs` from real conversations.
   - Add entropy-based pruning jobs so `tbl_l1_ngram_counts` stays within manageable bounds.
   - Experiment with adaptive stitching lengths to mimic GPT's dynamic stopping criteria.

This document should serve as the authoritative reference for anyone extending `src/db_slm`,
eliminating ambiguity about how data is stored, trained, and consumed at each level of the DB-SLM
stack.
