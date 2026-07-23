# lmdb
[Experimental] Database centric LLM

**Currently the project lacks totally of algorithmic optimizations by humans**

## Overview

This repository explores the production-grade database-native statistical language model (DB-SLM)
described in `studies/CONCEPT.md` and refined in
`studies/DB_SLM_DATABASE_AND_ALGORITHMS.md`. The Python implementation under `src/db_slm` now mirrors
the spec end-to-end:

- **Level 1 — Aria-style n-gram engine:** Byte-free (regex) tokenization backed by a relational
  vocabulary, hashed contexts, Modified Kneser–Ney smoothing, quantized log-prob tables, and a
  Top-K materialization path for fast sampling. SQLite owns the relational vocabulary and
  materialization workspace, while every context/count/probability/Top-K slice is streamed into
  Cheetah namespaces so Level 1 distribution reads can use the Go hot path.
  The decoder scoring pipeline now supports optional trace snapshots, making it easier to inspect
  base log10 values, penalties, cache blends, and prediction-table biasing when debugging output.
- **Level 2 — Episodic memory + biasing:** Conversation logging, correction digests, logit-bias
  materialization, and pointer-sentinel session caches that feed the decoder without tensors.
- **Level 3 — Concept model:** Concept dictionaries, templates, and probability tables that can
  output multi-token spans before the Level 1 stitcher runs.

Training stages materialization in SQLite, while production Level 1 contexts, reducers, Top-K
lookups, cache mixtures, and prediction signals are persisted and served through Cheetah.

## Quick Start

```python
from db_slm import DBSLMEngine

engine = DBSLMEngine()
conversation_id = engine.start_conversation(user_id="demo")
print(engine.respond(conversation_id, "Remind me what we discussed."))
```

Use `train_from_text()` to ingest corpora. It automatically updates counts, rebuilds the KN
probabilities, and refreshes the Top-K cache so the decoder can read quantized distributions directly
from the database.

## Environment Setup

- Python 3.10+ is recommended. Create an isolated virtual environment if you plan to experiment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- The CLI utilities still stage ingest output under `var/db_slm.sqlite3` so you can delete/reset runs
  quickly. Treat this as a scratch file: keep `DBSLM_BACKEND=cheetah-db` for real training/decoding,
  and only swap the SQLite path (`--db var/tmp.sqlite3`, `:memory:`, etc.) for small/fast exports.
- `.env` now defaults `DBSLM_BACKEND=cheetah-db`, so Level 1 lookups hit the Go service out of the
  box. SQLite remains available for bulk ingest/experiments (`DBSLM_BACKEND=sqlite`), but there is no
  longer a secondary relational target to keep in sync.

### cheetah-db runtime (required submodule)

- `cheetah-db/` is the [`cekkr/cheetah`](https://github.com/cekkr/cheetah) Git submodule and hosts
  the Go service that acts as the sole hot path for Level 1 contexts. Clone with
  `git clone --recurse-submodules https://github.com/cekkr/lmdb.git`, or initialize an existing clone with
  `git submodule update --init --recursive`. Build it with `bash cheetah-db/build.sh` and keep
  `./cheetah-db/cheetah-server` running before invoking any Python tooling.
- Export `CHEETAH_HEADLESS=1` when launching the server (e.g.
  `wsl.exe -d Ubuntu-24.04 -- screen -dmS cheetahdb bash -c 'cd /mnt/c/.../cheetah-db && env CHEETAH_HEADLESS=1 ./cheetah-server'`)
  to disable the interactive CLI and keep the TCP listener running in the background. Use the helper
  scripts (`scripts/start_cheetah_server.sh`, `scripts/stop_cheetah_server.sh`) for bounded,
  screen-first process management, exact PID tracking, and log rotation. They fall back to `tmux`
  only when `screen` cannot run on the host.
- Leave `DBSLM_BACKEND=cheetah-db` (the baked-in default) so Level 1 contexts and prediction signals
  use the Go hot path; vocabulary and relational Level 2/3 state still use SQLite. A configured
  Cheetah backend currently fails closed when its startup connection is unavailable. For an
  intentional reduced-feature SQLite experiment, set `DBSLM_BACKEND=sqlite` before invoking the
  trainer; do not rely on `--backonsqlite` to catch an unreachable configured Cheetah service until
  the startup fallback is repaired.
- The `DBSLM_CHEETAH_HOST/PORT/DATABASE/TIMEOUT_SECONDS` variables (see `.env.example`) point the
  adapter at the right instance; the default matches the server exposed by `cheetah-db/src/main.go`. Use
  a real address (127.0.0.1, LAN IP, Windows bridge IP inside WSL) rather than `0.0.0.0`.
- Idle responses are now capped at ~5 minutes on the Python side even when
  `DBSLM_CHEETAH_TIMEOUT_SECONDS` is raised for slow disks. Override
  `DBSLM_CHEETAH_IDLE_GRACE_SECONDS` for an explicit window or set
  `DBSLM_CHEETAH_IDLE_GRACE_CAP_SECONDS` (defaults to `300`) to clamp the derived value; set the cap
  to `0` to disable it. Heavy reducers now queue through the canonical `JOB submit
  command=<base64>` API, poll `JOB status id=...`, and retrieve the result once with `JOB fetch
  id=...`. The adapter uses `PAIR_REDUCE_ASYNC` / `PAIR_REDUCE_FETCH` only as compatibility aliases
  when an older server rejects `JOB`; it also removes the storage-transport base64 layer from
  reducer rows before decoding DB-SLM payloads. Tune `CHEETAH_REDUCE_ASYNC` (set to `0` for the
  synchronous reducer call) and
  `CHEETAH_REDUCE_POLL_INTERVAL_SECONDS` to change the cadence. While waiting, the adapter logs job
  state/percentage so stalled reducers are obvious in the trainer output.
- During ingest the Python pipeline streams new context metadata and Top-K probability slices into
  cheetah so the decoder can read them without re-querying SQLite, satisfying the adapter roadmap in
  `cheetah-db/README.md`.
- `PAIR_SCAN`/`PAIR_REDUCE` accept cursors and return `next_cursor=x...` when more data is available.
  The Python adapter follows these cursors automatically, so namespace walks and reducer projections
  can stream through arbitrary volumes of contexts without manual pagination.
- For deeper backend documentation, read `cheetah-db/README.md` (architecture and commands) alongside
  `cheetah-db/AGENTS.md` (operational checklists, cache budgets, and contributor rules). Generic
  server fixes belong in the upstream submodule, not in a vendored copy here. The current local
  gitlink includes upstream repair commit `6866be9`, which atomically reserves value slots and
  serializes full pair mutations to preserve concurrent ingest data.

#### Example `.env` block

```env
DBSLM_BACKEND=cheetah-db
DBSLM_SQLITE_PATH=var/db_slm.sqlite3
DBSLM_CHEETAH_HOST=127.0.0.1
DBSLM_CHEETAH_PORT=4455
DBSLM_CHEETAH_TIMEOUT_SECONDS=1.0
# Uncomment to extend the per-request idle grace (defaults to max(timeout*180, 60)s)
# DBSLM_CHEETAH_IDLE_GRACE_SECONDS=300
DBSLM_CHEETAH_IDLE_GRACE_CAP_SECONDS=300
CHEETAH_REDUCE_ASYNC=1
CHEETAH_REDUCE_POLL_INTERVAL_SECONDS=5
# Uncomment to select a named database/namespace on shared cheetah instances:
# DBSLM_CHEETAH_DATABASE=default
```

Copy `.env.example` to `.env`, adjust the host/port/database per deployment, and commit to keeping
`DBSLM_BACKEND=cheetah-db`. Override `DBSLM_CHEETAH_HOST` with the LAN/Windows bridge IP when the
server runs outside WSL; the adapter auto-detects that scenario via `/etc/resolv.conf`.

#### CLI + script knobs

- `scripts/start_cheetah_server.sh` / `scripts/stop_cheetah_server.sh` respect `CHEETAH_SERVER_BIN`,
  `CHEETAH_SERVER_SESSION`, `CHEETAH_SERVER_LOG`, `CHEETAH_SERVER_PID_FILE`, and
  `CHEETAH_SERVER_TIMEOUT` so you can pin the binary, session name, log/PID locations, and lifetime.
  The stop helper kills only the recorded process. Example:\
  `CHEETAH_SERVER_BIN=$PWD/cheetah-db/cheetah-server CHEETAH_SERVER_SESSION=cheetah-dev scripts/start_cheetah_server.sh`
- `scripts/run_cheetah_smoke.sh` spins up a short ingest/eval loop with cheetah as the backend; tune
  `CHEETAH_SMOKE_DATABASE`, `CHEETAH_SMOKE_MAX_JSON_LINES`,
  `CHEETAH_SMOKE_JSON_CHUNK_SIZE`, `CHEETAH_SMOKE_EVAL_INTERVAL`, `CHEETAH_SMOKE_DB`,
  `CHEETAH_SMOKE_TIMEOUT`, `CHEETAH_SMOKE_LOG`, or `CHEETAH_SMOKE_METRICS` to isolate the named
  Cheetah database and bound the dataset, evaluation cadence, scratch file, lifetime, and artifacts.
- `python src/train.py ... --reset` clears the SQLite scratch file **and** purges cheetah namespaces.
  The trainer now attempts `RESET_DB <DBSLM_CHEETAH_DATABASE>` first (instant file deletion), then
  falls back to `PAIR_PURGE` or the incremental scanner when connecting to older cheetah binaries.
  so cached Top-K slices never drift. For a deliberately degraded run without Cheetah (for example,
  an offline CI sandbox), set `DBSLM_BACKEND=sqlite` explicitly.

#### Verified emotion-data path

The repaired external submodule and adapter were validated together on 2026-07-23 using a named
Cheetah database and no SQLite fallback. A bounded 20-record run of
`datasets/emotion_data.json` ingested 380,200 tokens / 380,198 n-grams, completed 431 prediction
updates in 206.82 seconds, and a subsequent `run.py` process reused the same database for a correctly
framed inference response. See `studies/BENCHMARKS.md` for the exact commands, concurrency
regressions, storage summaries, and adaptive Go benchmark.

## Training CLI (`src/train.py`)

`train.py` is the canonical way to populate the Level 1 n-gram tables from plain-text corpora. Each
file is streamed into `DBSLMEngine.train_from_text()`, which updates hashed context counts, rebuilds
Modified Kneser–Ney statistics for every order up to the configured `--ngram-order`, refreshes
continuation counts, and re-materializes quantized probability tables plus the Top-K head cache.


### Example workflows

**Plain-text directories**

> Remember to install dictionaries:  
`$ python -m spacy download en_core_web_lg`  
`$ python -m spacy download en_core_web_sm`

```bash
python src/train.py \
  data/corpus.txt docs/*.txt \
  --db var/db_slm.sqlite3 \
  --ngram-order 0 \
  --recursive \
  --reset
```

**Chunked NDJSON ingest with live evaluation**

```bash
python src/train.py datasets/emotion_data.json \
  --db var/db_slm.sqlite3 \
  --ngram-order 0 \
  --json-chunk-size 750 \
  --chunk-eval-percent 12.5 \
  --eval-interval 40000 \
  --eval-samples 4 \
  --eval-variants 3 \
  --eval-dataset datasets/emotion_holdout.json \
  --metrics-export var/eval_logs/train-emotion.json \
  --decoder-presence-penalty 0.20 \
  --decoder-frequency-penalty 0.05 \
  --profile-ingest \
  --seed 1337
```

### Argument guide

- **Core ingest + storage**
  - `inputs`: Files or directories to ingest. Directories respect `--recursive` and only pull in `*.txt` files; explicit file arguments may be pre-tagged `.txt` corpora or `.json`/`.ndjson` datasets that pick up their configs automatically.
  - `--db`: Destination SQLite file. Parent directories are created automatically; use `:memory:` for scratch runs (cannot be combined with `--reset`). Keep the chosen path consistent with `run.py`.
  - `--reset`: Delete the existing database before ingesting so you start from a clean slate.
  - `--backonsqlite`: Reserved for an explicit SQLite fallback, but the current adapter exits during
    startup before this branch can handle an unreachable configured Cheetah backend. Use
    `DBSLM_BACKEND=sqlite` for an intentional SQLite-only run until the fallback path has a startup
    regression test.
  - `--ngram-order`: Adjusts the context window length (use `0` for auto-selection based on a corpus sample). Higher orders need larger corpora but produce richer continuations.
  - `--merge-max-tokens`: When `--ngram-order` is 5 or higher, merge repeated token runs (up to `merge-max-tokens`, default 5) into composite vocabulary entries, then optionally recurse across the merged stream. Only spans at or above the average frequency of all candidate spans survive, and runs dominated by high-frequency tokens are down-weighted so generic phrases are less likely to merge. Set `--merge-max-tokens 0` to disable. Composite tokens inherit cheetah prediction weights via batched PREDICT_INHERIT jobs during training.
  - `--merge-recursion-depth`: Recursive merge passes to attempt per tokenization step (defaults to 2 when merging is enabled).
  - `--merge-train-baseline` / `--no-merge-train-baseline`: Train the unmerged token sequence alongside merged tokens (defaults to enabled when merging is active).
  - `--merge-eval-baseline` / `--no-merge-eval-baseline`: Log perplexity metrics with merging disabled for comparison (defaults to enabled when merging is active).
  - `--merge-significance-threshold`: Retire merge tokens whose applied/candidate ratio falls below the threshold (0 disables).
  - `--merge-significance-min-count`: Minimum candidate count before evaluating merge significance (default 2).
  - `--merge-significance-cap`: Cap how many retired merge tokens are persisted in metadata (default 128).
  - `--context-dimensions "<ranges>"`: Extends repeat penalties across grouped token spans (e.g., `1-2,3-5` or progressive lengths like `4,8,4`). Use presets `default`/`deep`/`shallow`, or `off`/`none` to disable. Selections persist in `tbl_metadata` and the cheetah metadata mirror.
  - `--context-window-train-windows <n>`: Override the cap for adaptive windows-per-dimension sampling during training for context embeddings (0 = auto).
  - `--context-window-infer-windows <n>`: Override how many windows per dimension are sampled during inference/evaluation for context embeddings (0 = auto).
  - `--context-window-stride-ratio <float>`: Override the window stride ratio used for context embeddings (0.1-1.0, 0 = auto).
  - `--context-window-depth <n>`: Bias extra context-matrix fusion depth tiers (default engine preset). Use `0` to match legacy depth, negative values reduce depth.
  - `--dataset-config <path>`: Force a specific dataset metadata/label file for `.json`/`.ndjson` corpora instead of inferring `<dataset>.config.json` or honoring `DBSLM_DATASET_CONFIG_PATH`. Plain `.txt` corpora bypass this path and are treated as already tagged.
  - `--sentence-splitting` / `--no-sentence-splitting`: Enable punctuation-based sentence segmentation during training (disabled by default; set `DBSLM_SENTENCE_SPLIT=1` to change the default).
- **File reading helpers**
  - `--recursive`: When scanning folders, include subdirectories (default is to read only the top level).
  - `--encoding`: Override the UTF-8 reader if the corpus uses another encoding.
  - `--stdin`: Stream additional ad-hoc text from `STDIN`, e.g. `cat notes.txt | python src/train.py --stdin`.
- **JSON / NDJSON streaming + hold-outs**
  - `--json-chunk-size`: Stream JSON/NDJSON rows in fixed-size batches so memory stays bounded.
  - `--max-json-lines`: Limit how many JSON rows load per file when you just need a smoke test.
  - `--chunk-eval-percent`: Reserve this percentage of every JSON chunk as an immediate evaluation set. Hold-outs run through the inference stack before the chunk trains and refresh the probe pool.
  - `--seed`: Seed Python's RNG for deterministic chunk sampling, hold-outs, and paraphraser tweaks.
- **Evaluation cadence + randomness**
  - `--eval-interval <tokens>`: Trigger periodic probes every N ingested tokens (0 disables the loop).
  - `--eval-samples <count>`: Held-out prompts per probe (minimum 2, default 3).
  - `--eval-variants <count>`: Responses per prompt (defaults to 2 when context dimensions are enabled, otherwise 1) so you can compare structural diversity.
  - `--eval-seed <int>`: Base seed for evaluation randomness. Each prompt/variant gets a deterministic sub-seed derived from this value.
  - `--eval-dataset <path>`: NDJSON file containing `prompt`/`response` pairs; defaults to `DBSLM_DATASET_PATH` from `.env`.
  - `--eval-dataset-config <path>`: Override the metadata/label mapping used for `--eval-dataset`. Falls back to `<dataset>.config.json` or `DBSLM_DATASET_CONFIG_PATH` when omitted.
  - `--eval-pool-size <count>`: Maximum number of records kept in memory for the rolling evaluation pool (default 200, 0/None means unlimited).
- **Profiling + logging**
  - `--profile-ingest`: Print per-corpus latency and RSS metrics while ingesting so you can raise chunk sizes confidently.
  - `--metrics-export <path>`: Write the rolling ROUGE/perplexity timeline plus profiling samples to JSON (`var/eval_logs/train-<timestamp>.json` by default). Use `--metrics-export -` to disable.
- **Decoder penalty overrides (evaluation-only)**
  - `--decoder-presence-penalty <float>`: Adds a one-time penalty when a token/span has already appeared in the generation. Typical sweeps cover `0.0-0.4`.
  - `--decoder-frequency-penalty <float>`: Scales penalties by how often the token/span repeats. Values between `0.0` and `0.2` usually smooth repetition without collapsing the sampler.

Every run reports per-file token counts, derived n-gram windows, and the evaluation log path. Inputs shorter than the configured order are skipped automatically and clearly labeled in the logs.
Running `python src/train.py` with no arguments resumes the last interrupted training run using `var/train_resume.json`, skipping any completed chunks. Pass explicit inputs to start a fresh ingest.

#### Dependency parsing layer & strong token groups

- Every JSON/NDJSON row now runs through a dependency parser (spaCy first, Stanza as a fallback) when
  preparing training/evaluation samples. The resulting arcs and categorical buckets are appended to
  each segment as a `DependencyLayer: {...}` line so the downstream n-gram prep can treat those terms
  as a strong reference set. This extra layer gives the hold-out sampler the "important tokens" with
  far fewer n-gram windows than a naïve surface-form scan.
- The serialized payload includes the backend (`spacy` or `stanza`), the flattened dependency arcs,
  and a `strong_reference` map that classifies lemmas into buckets such as `subjects`, `objects`,
  `actions`, and `modifiers`. Evaluations report two additional metrics derived from the same data:
  `strong_token_overlap` (share of critical lemmas preserved) and `dependency_arc_overlap` (matching
  head/dependency triples). Both values surface in the probe summaries next to ROUGE/perplexity.
- Configure the parsers via environment variables: `DBSLM_SPACY_MODEL` (default `en_core_web_sm`),
  `DBSLM_DEP_LANG` (default `en` for Stanza), and `DBSLM_STANZA_PROCESSORS`
  (default `tokenize,pos,lemma,depparse`). When neither backend is installed the trainer logs a
  single warning and continues without the layer.
- Remember to install at least one backend model before training: e.g.
  `python -m spacy download en_core_web_sm` or `python -c "import stanza; stanza.download('en')"`. The
  base `requirements.txt` already lists both libraries so `pip install -r requirements.txt` pulls the
  Python packages automatically.

Large batches can take a while to finish a single call to `train_from_text()`, so the trainer now
prints progress lines for the vocabulary pass, every n-gram order, and the KN rebuilds. The logs
include an approximate row number so you can tell which part of the chunk is currently in flight
instead of staring at a silent terminal.

Example (quick validation run that ingests only 200 lines and probes the decoder every ~2k tokens):

```bash
python3 src/train.py datasets/emotion_data.json \
  --db var/db_slm.sqlite3 \
  --reset \
  --json-chunk-size 100 \
  --max-json-lines 200 \
  --eval-interval 2000 \
  --eval-samples 2 \
  --eval-pool-size 20
```

### Dataset defaults & configs

`DBSLM_DATASET_PATH` in `.env` points at the canonical NDJSON dataset used for evaluation seeds
(`datasets/emotion_data.json` by default). When you pass any `.json`/`.ndjson` corpus to
`src/train.py`, the loader automatically searches for `<dataset>.config.json` (next to the data) or
the config supplied via `--dataset-config`/`DBSLM_DATASET_CONFIG_PATH`. That file declares the
prompt/response labels and optional context fields so the trainer can preserve each `|TAG|:` prefix
as an atomic token and, when a context field sets `canonical_tag`, emit the canonical `|CTX|:` (or
custom) headers for you. Because the lookup runs per file, you can mix multiple JSON corpora in a
single command—each will read its sibling `.config.json` unless you explicitly override the path.
Plain `.txt` inputs continue to work for already-tagged corpora—they skip config discovery entirely
and stage the text as-is.

Use `--eval-dataset` plus `--eval-dataset-config` when the hold-out file differs from the training
corpora. Otherwise the evaluator inherits the same discovery rules as training (infer, then fall back
to the environment variable).

### Adaptive Tokenization + Context Tags

`DBSLMEngine` can run an optional realtime corpus scan before ingest to discover productive
punctuation splits, slice long responses into manageable segments, and tag those fragments with
device-aware embeddings from `sentence-transformers` (defaults to `all-MiniLM-L6-v2`, configurable
via `DBSLM_EMBEDDER_MODEL`). This punctuation splitting is disabled by default; enable it with
`--sentence-splitting` or `DBSLM_SENTENCE_SPLIT=1` if you need the legacy segmentation pass.
Dataset-specific metadata is described in
`datasets/<name>.config.json` (for example `datasets/emotion_data.config.json`), which declares the
prompt/response fields and any additional context columns that should be tokenized. The JSON loader
uses that config to emit human-readable headers plus canonical `|CTX|:<token>:<value>` (or other
`canonical_tag`) lines whenever a context field requests them, and the embedding pipeline lifts
those tags into a unified header before sequencing the text. Each
segment keeps a compact embedding signature plus a `|CTX_KEY|` keyword list derived from both the
dataset metadata and the embedding energy. These annotations are injected ahead of the regex
tokenizer, ensuring the vocabulary learns explicit dataset context and higher quality boundary
splits even while the underlying Level 1 tables remain purely relational. When the optional
dependency is missing, the pipeline falls back to deterministic hashed vectors so tokenization still
benefits from the dataset profiler. To avoid Hugging Face downloads entirely (e.g., in CI or
air-gapped labs), set `DBSLM_EMBEDDER_OFFLINE=1` or choose `DBSLM_EMBEDDER_MODEL=hashed` and the
hashed guidance path is used from the start.

### Instruction and Response Tags

Prompt/response scaffolding now supports the explicit tags that the evaluator and regression tests
already expect. `train.py` prints whatever `prompt_label`/`response_label` the dataset config
provides, and any `context_fields` entry can opt into `"placement": "before_prompt"` so its label is
inserted immediately ahead of the user prompt. `DatasetConfig.compose_prompt()` mirrors those
preface lines when building held-out prompts, letting evaluation probes replay the exact
`|INSTRUCTION|` + `|USER|` framing produced during staging. Interactive helpers (`run.py`, the
evaluation probes, and the paraphraser guards in `src/db_slm/pipeline.py`) wrap prompts with
`|USER|:` and model outputs with `|RESPONSE|:` so downstream tooling can distinguish user turns from
generations. Every response sent through the trainer also flows through `append_end_marker()`,
guaranteeing the sentence-level `|END|` marker is present even if a dataset omits it. The tokenizer
treats `|END|` like the other structural markers, so appending it does not change semantic content
but keeps segment boundaries unambiguous.

Candidate decoding rejects prompt-tag aliases up to the bounded retry budget. If all attempts remain
contaminated, the engine discards their token IDs and produces a scaffold-free response backstop
instead of leaking the final invalid candidate into the response or cache. The tagged formatter also
strips an existing terminal `|RESPONSE|:` before wrapping a prompt, preventing nested `|USER|:`
frames in scripted inference.

`db_slm.prompt_tags.ensure_response_prompt_tag()` is now called by both `train.py` and the
evaluation stack immediately before decoding so prompts always terminate with the configured
response label (default `|RESPONSE|:`). `run.py` exposes `--response-label` for interactive sessions
and uses the same helper. This is a high-priority invariant: without the sentinel the decoder will
happily continue the `|USER|:` frame instead of predicting the reply, corrupting both training and
evaluation logs.
Prompt-tag bans and evaluation detection normalize case when `DBSLM_TOKENIZER_LOWERCASE=1`, so
`|RESPONSE|:` or `|CONTEXT|:` markers stay blocked even if tokens are lowercased during decoding.

When crafting new dataset configs place the canonical labels directly in the JSON. The updated
`datasets/GPTeacher.config.json` file now maps `input` → `|USER|` and registers the `instruction`
column as a context field with `"placement": "before_prompt"`, yielding staged samples that begin
with ``|INSTRUCTION|: ...`` followed by the actual ``|USER|: ...`` prompt. No code changes are
required when adding similar datasets—`load_dataset_config()` already respects arbitrary labels,
tokens, and placements supplied by the config file.
Evaluation datasets still require the real prompt column; optional instruction/context fields are
only added when present in the source JSON.

### cheetah Streaming Archive

The trainer streams every newly discovered context plus its Top-K probability slices directly into
cheetah-db as part of the ingest loop.
`DBSLM_BACKEND=cheetah-db` keeps the decoder entirely on the Go engine, so Level 1 lookups never
round-trip through SQLite. The adapter now tracks cache coverage
(`DBSLMEngine.cheetah_topk_ratio()`) and exposes namespace scanners plus the new
`engine.context_relativism([...])` helper for probabilistic trie queries. Every context mirrored into
cheetah also gets an Absolute Vector Order signature (`ctxv:` namespace) so nested context requests
([[token arrays], ...]) are deterministically sorted and can be re-hydrated via `PAIR_SCAN` alone.
`Decoder` now falls back to those relativistic slices whenever cheetah misses a direct Top-K entry.
MKNS rebuilds mirror raw follower counts through the new `PAIR_REDUCE counts` RPC, so server-side
reducers stream the context registry straight from Go and delete the last SQLite-only temporary
tables. SQLite only keeps a scratch copy for bulk rebuilds, so there is no secondary database to
drain—cheetah already holds the hot/archived copies in one place. For namespace triage, cache sizing
tables, and screen-first launch recipes, consult `cheetah-db/README.md` and `cheetah-db/AGENTS.md`.

### Training-Time Metrics

Every evaluation probe now prints lexical-overlap, ROUGE-L, and a perplexity stub for both the
generated response and the held-out reference. This lets you confirm that quantitative scores improve
while you experiment with chunk sizes or smoothing tweaks, even when the qualitative samples look
similar. When `--chunk-eval-percent` is supplied, the same metric stack runs immediately after each
chunk ingest using its freshly carved-out hold-outs, giving you a rolling measure that tracks the
latest corpus segment instead of relying solely on the static evaluation dataset. Probes start a
fresh conversation with the low-resource seeding helper disabled, so the reported generations will
reflect the newly trained n-gram tables instead of the caretaker seed dialog you may see in
interactive `run.py` sessions on tiny corpora.

In addition to the per-sample logs, `train.py` now prints run-level averages (lexical overlap,
ROUGE-L, generated/reference perplexity) after every probe and mirrors the raw samples into
`var/eval_logs/*.json`. The feed includes hold-out probes, periodic evaluation sets, and optional
profiling records so you can diff long runs or export the JSON into your own dashboards. Point
`--metrics-export` at a custom path when you need to archive the file elsewhere.

Evaluation probes also call the new sentence-quality stack: LanguageTool for grammar deltas,
`textattack/roberta-base-CoLA` for semantic acceptability, and the shared sentence-transformer
embedder for similarity/novelty scores. Those numbers are appended to the JSON timeline alongside
lexical/ROUGE/perplexity values, so you can catch regressions that only manifest as grammatical
errors or semantic drift. Because `emotion_data.json` responses average ~347 words, the evaluator
derives `min_response_words` from the reference length (capped at 512 words) to ensure the logged
`|RESPONSE|` frame actually reaches the substantive part of the answer instead of truncating after
128 words. Each prompt/variant now receives a dedicated RNG seed derived from `--eval-seed` (or a
per-run random base when the flag is omitted), so repeated prompts explore different structures
without manually juggling randomness. When a sample is flagged for retraining it now re-enters the
current batch at a random
position (up to two total attempts) before being scheduled for future probes, so the decoder gets a
fresh shot without holding up the rest of the evaluation.

Set `DEVICE=cuda` or `DEVICE=mps` before launching `src/train.py` to force the sentence-transformer
embedder and CoLA classifier used by the evaluation stack onto that accelerator whenever PyTorch
reports it as available. Unsupported requests automatically fall back to CPU and emit a single
notice so the run keeps going.

Supplying `--decoder-presence-penalty` or `--decoder-frequency-penalty` only affects the inference
path used by those probes (including chunk hold-outs); training statistics stay unchanged. When the
flags are omitted, `train.py` auto-tunes the penalties after each probe/hold-out using the repetition
and structure metrics, while explicit values lock those knobs. The chosen values flow into the
`DecoderConfig` passed to `issue_prompt()` and are emitted in the metadata block inside
`var/eval_logs/*.json`, so repeat-penalty sweeps can be compared later without scraping the console
logs.

Low-quality generations (grammar errors ≥ 3, CoLA < 0.45, semantic similarity < 0.55, or a >40%
length mismatch) are streamed into `DBSLM_QUALITY_QUEUE_PATH` (defaults to
`var/eval_logs/quality_retrain_queue.jsonl`). This “retrain queue” doubles as a regression fixture:
drop the file back into `train.py` as an evaluation dataset and the weakest samples receive targeted
attention during the next ingest. Heavy grammar/semantic scoring only runs when the adaptive CPU
guard detects spare headroom, so long streaming ingests do not pay a latency penalty on saturated
laptops.

### Queue-Drain Automation

Use `scripts/drain_queue.py` when the retrain queue approaches 150 entries. The helper inspects
`DBSLM_QUALITY_QUEUE_PATH`, runs the documented queue-drain preset via `python3.14 src/train.py ...`,
forces `--max-json-lines 500` to keep chunk sizes consistent during tests, and trims the queue back
to `--queue-cap` entries (defaults to 200) after a successful drain. Metrics land in
`var/eval_logs/train-queue-drain-*.json`; throughput and the metrics path are echoed to stdout so you
can log the cleanup inside `studies/BENCHMARKS.md`. `--threshold`, `--max-json-lines`, and
`--queue-cap` are tunable, `--python` lets you point at a custom interpreter, and `--dry-run`
previews the exact command (useful when orchestrating via
`wsl.exe -d Ubuntu-24.04 -- PYTHONPATH=src ... scripts/drain_queue.py ...`).

### Cheetah Prediction Tables & Context Probes

The Python adapter now exposes cheetah's prediction-table commands, letting you exercise the context
matrices described in `cheetah-db/README.md` without leaving the CLI:

- `ContextWindowEmbeddingManager.context_matrix_for_text()` now emits the per-window vectors plus
  dimension-level summary/fusion layers so prediction tables see a hidden-layer style context
  matrix aligned with `--context-dimensions`. The Python side auto-adds extra fused tiers when
  dimension summaries diverge; `--context-window-depth` biases how deep those hidden layers run.
  cheetah-db further deepens these matrices with derived mean/variance/contrast/interaction layers
  that scale with context diversity during
  training/querying (disable with `CHEETAH_PREDICT_DEEPEN=0`). `train.py` uses this helper whenever
  you pass one or more
  `--cheetah-context-probe "text snippet"` arguments (repeatable). Each snippet is converted into
  a context matrix via the active `--context-dimensions`, then piped through
  `PREDICT_QUERY table=context_matrices key=meta:context_dimension_embeddings` by default.
- Override `--cheetah-predict-table` or `--cheetah-predict-key` to target a different prediction
  shard, e.g. `--cheetah-predict-table ctx_predictions` for custom namespaces.
- Results are rendered with `helpers.cheetah_cli.format_prediction_query()`, so the log stream lists
  the backend, total hit count, and the top entries (`value_hex -> probability`) for inspection.
- Trainers now mirror prompt/response pairs into cheetah prediction tables in real time. JSON
  datasets feed the prompt text + dependency summaries through the context-window embedder, seed the
  next-token entry with `PREDICT_SET`, and update weights via `PREDICT_TRAIN`. Tune the behavior with
  `--cheetah-token-table`, `--cheetah-token-key`, `--cheetah-token-max-tokens`,
  `--cheetah-token-learning-rate`, and `--cheetah-token-value-cap`, or opt out entirely with
  `--disable-cheetah-token-train`.
- Evaluation failures now trigger adversarial cheetah updates by default. When a probe is flagged or
  its `quality_score` drops below `--cheetah-adversarial-threshold`, the trainer derives a context
  matrix from the prompt/metadata, reinforces the reference tokens, and down-weights the generated
  tokens in one `PREDICT_TRAIN` call (`negatives=`). Control the cadence with
  `--disable-cheetah-adversarial-train`, `--cheetah-adversarial-max-negatives`, and
  `--cheetah-adversarial-learning-rate` (defaults to 60% of the main cheetah-token rate).
- Decoding blends those predictions back into sampling. `train.py` probes and `run.py` both honor
  `--cheetah-token-table` / `--cheetah-token-key` and mix the cheetah probabilities using
  `--cheetah-token-weight` (defaults to `0.25`). This keeps next-token hints synchronized between
  training, evaluation, and the interactive REPL without extra wiring.
- Supply `--cheetah-eval-predict` during training to stream a prediction query for every evaluation
  sample. `--cheetah-eval-predict-source dependency` (default) converts the stored dependency layers
  into probe text, but `prompt`, `response`, `generated`, and `context` sources are also available;
  use `--cheetah-eval-predict-limit` to control the number of entries logged per sample.
- `run.py` mirrors the same wiring via `--cheetah-predict-log`. Interactive sessions can now show
  the prediction-table response derived from the conversation history (`--cheetah-predict-source
  history`), the user prompt, or the latest assistant response after every turn. The worker process
  issues the TCP command and streams the formatted lines back to the CLI so single-shot prompts
  (`--prompt`) and REPL turns share the same context-probability diagnostics.
- Example (run before ingest to inspect the current context-matrix state):
  ```bash
  python src/train.py datasets/emotion_data.json \
    --cheetah-context-probe "Summaries about remote work" \
    --cheetah-context-probe "Reflect on the previous lesson" \
    --cheetah-system-stats
  ```
  When context windows are disabled (e.g., `--context-dimensions off`), the trainer skips the probes
  automatically.

## Inference CLI (`src/run.py`)

`run.py` spins up a conversational REPL backed by the database produced during training. The loop
invokes the full decoding pipeline: Level 3 concept prediction (with signal overrides), Level 2 bias
and cache adjustments, and Level 1 top-p decoding with quantized probabilities.

Interactive session:

```bash
python src/run.py --db var/db_slm.sqlite3
[run] Using conversation: 6f5080d1-...
you> summarize our discussion
assistant> Based on our recent exchange: ...
you> :history   # prints the Level 2 context window
you> :exit
```

Single-shot inference (headless prompt):

```bash
python src/run.py --db var/db_slm.sqlite3 \
  --prompt "Remind me what we covered." \
  --context-dimensions off \
  --user qa-demo \
  --agent curator-bot
```

Scripted resumptions / limited-turn demos:

```bash
python src/run.py \
  --db var/db_slm.sqlite3 \
  --conversation 6f5080d1-... \
  --context-dimensions "1-2,4-6" \
  --max-turns 3
```

Key arguments:

- `--db`: SQLite file produced by `train.py`. Paths are created on demand, but `:memory:` is rejected so runs always persist conversation history.
- `--ngram-order`: Should match the value used during training; set to `0` to reuse the order stored in metadata (default behavior).
- `--context-dimensions`: Same parser as `train.py`. Override span ranges ("1-2,4-6", `off`, etc.) when you want to force different repeat penalties than the metadata stored alongside the database.
- `--prompt`: Skip the REPL and emit a single response (great for CI hooks or quick sanity checks). When omitted, interactive mode starts.
- `--instruction`/`--instruction-label`: Provide a system or teacher instruction that should always precede the real prompt. The CLI emits the block as ``|INSTRUCTION|: ...`` by default, matching the dataset framing.
- `--user-label`: Labels every prompt before it is handed to DBSLM (defaults to `|USER|`). Pass `--user-label ''` to opt out when a raw prompt is desired.
- `--conversation`: Resume a Level 2 conversation ID already stored in `tbl_l2_conversations`; omit to start a fresh session (the new ID is printed at startup).
- `--user` / `--agent`: Customize the identifiers written to Level 2 so parallel sessions stay distinguishable in the logs.
- `--max-turns`: Auto-exit after the specified number of user prompts. Useful when scripting deterministic walkthroughs.
- REPL commands: `:history` prints the Level 2 context window, `:exit`/`:quit` (or `Ctrl+D`) leaves the session immediately.

Level 3 context summaries (`ContextSummary` signals) still influence decoding, but their `|CONTEXT|:` scaffolding
never surfaces in responses. The engine injects those payloads solely into the rolling bias text and context-dimension
weights so datasets remain in full control of which structural tags appear in prompts or completions.

Because the CLI uses the exact same engine object, anything logged via `run.py` is immediately
available to downstream tooling (correction logging, concept payload providers, etc.).

Small validation runs sometimes overfit; the engine now seeds each conversation with short caretaker
exchanges and paraphrases overly similar replies so you still get meaningful summaries instead of a
verbatim echo. Multi-turn prompts and corrective instructions are explicitly guarded, so the
paraphraser never rewrites structured guidance or follow-up directions.

`scripts/run_paraphraser_regression.py` exercises those guard rails against
`studies/paraphraser_regression.jsonl`, which mixes multi-turn corrective threads, structural tags,
and plain prompts that should still be rewritten. Wire it into CI or run it locally whenever you
tweak `SimpleParaphraser`.

Training-time evaluations were further hardened so the decoder always produces at least 20 words,
even when the probabilistic backoff is uncertain. The new response backstop adds transparent filler
sentences referencing the prompt keywords so ROUGE/perplexity measurements never silently drop rows.

## Smoke Testing

`make smoke-train` now shells into `scripts/smoke_train.py`, which executes a configurable matrix of
scenarios in series while streaming their progress into `var/smoke_train/benchmarks.json`. The
default suite contains two complementary runs:

- `baseline_profiled` reproduces the original capped ingest (400 NDJSON rows, profiling enabled) and
  keeps the existing probe cadence.
- `penalty_sweep_holdout` trims ingest to 240 rows, enables chunk hold-outs, and overrides the
  decoder penalties/context-dimension spans to stress the repetition guards.

Each scenario receives its own SQLite file plus a dedicated `DBSLM_CHEETAH_DATABASE` namespace, so
the Go service can be pointed at different logical stores without restarting. The script tails the
trainer stdout, updates the benchmark JSON with progress percentages, token counts, and the latest
log lines, and drops the full evaluation payload for every run under
`var/smoke_train/metrics/<scenario>.json`. External agents (or another terminal) can watch the
benchmark file to decide when to halt, resume, or reprioritize scenarios in real time.

Key flags:

- Limit the matrix: `make smoke-train SMOKE_SCENARIOS=baseline_profiled` (comma-separated names) or
  `scripts/smoke_train.py --scenarios baseline_profiled,penalty_sweep_holdout`.
- Feed a custom JSON matrix: `make smoke-train SMOKE_MATRIX=studies/my_smoke_matrix.json`.
- Preview without running anything: `scripts/smoke_train.py --dry-run`.
- Override the benchmark location: `make smoke-train SMOKE_BENCH=var/smoke_train/custom.json`.

Use `make clean-smoke` to delete the per-scenario SQLite files and the `var/smoke_train` artifacts.

## SQLite helpers

SQLite remains the relational workspace for the vocabulary, schema materialization, conversations,
biases, and concepts; use disposable paths for fast ingest experiments, CI smoke runs, or quick
exports where deleting the file to reset state is desirable. WAL keeps throughput high for those
runs, while cheetah-db mirrors Level 1 contexts, counts, probabilities, continuations, and Top-K
buckets for hot distribution reads. There is no MySQL target or automated migration path. Preserve
both the SQLite file and the selected Cheetah database when a full trained/conversational state must
survive. Run the Go server, point the env vars at it, and use helpers such as
`engine.iter_hot_context_hashes()` or
`engine.context_relativism(...)` when you need ordered scans or probabilistic tree walks over the
stored contexts. Delete or vacuum SQLite only when its relational state is explicitly disposable.
