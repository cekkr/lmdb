# DB-SLM (`lmdb`)

DB-SLM is an experimental, database-native statistical language model. It trains n-gram
distributions from text or structured conversation datasets, stores the hot Level 1 model in
[Cheetah](https://github.com/cekkr/cheetah), and adds persistent conversation memory, contextual
biases, concept templates, evaluation, and prediction signals in Python.

This repository is research software:

- It is not the Lightning Memory-Mapped Database library, despite the repository name.
- It is not a transformer-based LLM and does not call one to generate tokens.
- It is not production-ready, and current experiments do not establish competitive generation
  quality or scale.

The supported workflow is Cheetah-first. SQLite is still used internally as the relational companion
store for vocabulary, training materialization, conversations, biases, and concepts, but SQLite-only
training is a reduced research mode and is intentionally not used in this guide.

## How DB-SLM works

DB-SLM has three cooperating levels:

1. **Level 1 — statistical generation:** tokenization, n-gram counts, Modified Kneser–Ney
   probabilities, quantized Top-K continuations, and Cheetah prediction tables.
2. **Level 2 — conversational memory:** persisted messages, corrections, session caches, and
   contextual token biases.
3. **Level 3 — concepts:** concept probabilities and templates that can guide the Level 1 decoder.

An optional fourth signal, **graph context memory**, records the corpus as entities and relations in
Cheetah's graph store and recalls them before decoding. It is off by default; see
[Graph context memory](#graph-context-memory).

Core generation always comes from the stored statistical model. Sentence Transformers, spaCy,
Stanza, LanguageTool, and the optional CoLA classifier provide context or evaluation signals; they
do not replace the database-native decoder.

A usable model instance consists of two coordinated stores:

- the named Cheetah database selected by `DBSLM_CHEETAH_DATABASE`;
- the relational companion file selected by `DBSLM_SQLITE_PATH`.

Training and inference must use the same pair. Preserve both when you want to keep a trained model
and its conversation state.

## Requirements

- Python 3.10 or newer
- Go 1.24 or newer to build Cheetah
- Java available on `PATH`; training validates it because LanguageTool is required at startup
- `screen` or `tmux` for the bounded server and long-running training helpers
- enough local storage for the corpus, Cheetah database, relational companion, metrics, and model
  caches

The first use of LanguageTool, Sentence Transformers, Stanza, or CoLA may download model data.

## Install

Clone the repository with its Cheetah submodule:

```bash
git clone --recurse-submodules https://github.com/cekkr/lmdb.git
cd lmdb
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Create the Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency parsing is optional, but installing one English model gives training and evaluation
stronger structural signals:

```bash
python -m spacy download en_core_web_sm
```

Copy the tracked configuration template:

```bash
cp .env.example .env
```

`.env` is ignored by Git. For a first isolated model, review at least these values:

```env
DBSLM_BACKEND=cheetah-db
DBSLM_CHEETAH_HOST=127.0.0.1
DBSLM_CHEETAH_PORT=4455
DBSLM_CHEETAH_DATABASE=emotion_demo
DBSLM_SQLITE_PATH=var/emotion_demo.sqlite3
```

Use a concrete client address such as `127.0.0.1`, not `0.0.0.0`. Cheetah's TCP protocol has no
authentication or TLS, so keep it on loopback or another trusted network.

## Build and run Cheetah

Build the pinned server:

```bash
bash cheetah-db/build.sh
```

Start it through the bounded process helper:

```bash
scripts/start_cheetah_server.sh
```

The helper starts a headless server in `screen`, falls back to `tmux`, records the exact PID, writes
the log to `var/cheetah-server.log`, and stops the process after 1,800 seconds by default. Inspect it
with:

```bash
tail -f var/cheetah-server.log
```

For a planned training segment that needs a longer server window, set an explicit bound before
starting it:

```bash
CHEETAH_SERVER_TIMEOUT=3600 scripts/start_cheetah_server.sh
```

Stop only the owned server session:

```bash
scripts/stop_cheetah_server.sh
```

Do not rely on `--backonsqlite` when Cheetah is unavailable. The configured Cheetah backend
currently fails closed during engine startup before that fallback can run. Check the server log,
host, port, and database name instead.

## Prepare a dataset

Raw datasets are deliberately not committed. Download or create the corpus locally and keep its
schema description next to it:

```text
datasets/emotion_data.json
datasets/emotion_data.config.json
```

The repository includes configs for:

- [`datasets/emotion_data.config.json`](datasets/emotion_data.config.json), which maps
  `prompt`/`response` and an `emotion` context field;
- [`datasets/GPTeacher.config.json`](datasets/GPTeacher.config.json), which maps `input`/`response`
  and places the `instruction` field before the user prompt.

Dataset sources are listed in [`datasets.md`](datasets.md).

JSON and NDJSON configs control the field mapping and framing used by both training and evaluation.
A minimal custom config looks like:

```json
{
  "name": "support_qa",
  "prompt_field": "question",
  "prompt_label": "|USER|",
  "response_field": "answer",
  "response_label": "|RESPONSE|",
  "context_fields": [
    {
      "field": "topic",
      "label": "Topic",
      "token_name": "topic",
      "canonical_tag": "|CTX|"
    }
  ]
}
```

Save it as `support_qa.config.json` beside `support_qa.json` or pass it explicitly with
`--dataset-config`. Plain `.txt` files are also accepted, but they bypass dataset-config discovery
and are ingested as already-framed corpus text. If a dataset uses nonstandard prompt or response
labels, pass the same labels to inference with `--user-label` and `--response-label`.

## Train a model

### First bounded training run

This command exercises the full Cheetah-backed ingest path on 20 emotion records without running
periodic evaluation:

```bash
PYTHONPATH=src python3 src/train.py datasets/emotion_data.json \
  --ngram-order 3 \
  --json-chunk-size 20 \
  --max-json-lines 20 \
  --eval-interval 0 \
  --max-runtime-seconds 900 \
  --profile-ingest \
  --metrics-export var/eval_logs/emotion-demo.json
```

It uses the Cheetah database and relational companion configured in `.env`. A 20-record slice is a
pipeline validation, not a useful language model; inference quality will remain limited.

To discard that isolated model and repeat from a clean state, add `--reset`:

```bash
PYTHONPATH=src python3 src/train.py datasets/emotion_data.json \
  --ngram-order 3 \
  --json-chunk-size 20 \
  --max-json-lines 20 \
  --eval-interval 0 \
  --metrics-export var/eval_logs/emotion-demo-reset.json \
  --reset
```

`--reset` is destructive across both configured stores. Verify
`DBSLM_CHEETAH_DATABASE` and `DBSLM_SQLITE_PATH` before using it, especially when the Cheetah server
is shared.

### Evaluation-enabled training

Treat this as an alternative run: select a fresh model pair or reset the isolated quick-start model
before it so the first 20 records are not counted twice. The command reserves part of each chunk for
immediate probes and runs periodic held-out evaluation:

```bash
PYTHONPATH=src python3 src/train.py datasets/emotion_data.json \
  --ngram-order 3 \
  --json-chunk-size 50 \
  --max-json-lines 250 \
  --chunk-eval-percent 4 \
  --eval-interval 100000 \
  --eval-samples 2 \
  --eval-variants 1 \
  --eval-pool-size 50 \
  --max-runtime-seconds 1200 \
  --profile-ingest \
  --metrics-export var/eval_logs/emotion-evaluated.json
```

Long runs should be launched in a bounded `screen` or `tmux` session. One straightforward workflow
is:

```bash
screen -S dbslm-train
```

Run the training command inside that session, then detach with `Ctrl+A`, `D`. Reattach with:

```bash
screen -r dbslm-train
```

### Pause and resume

`--max-runtime-seconds` finishes the active chunk and records a recoverable pause in
`var/train_resume.json`. Resume the saved command with no arguments:

```bash
PYTHONPATH=src python3 src/train.py
```

Resume requires the same input files and model configuration. Input size or modification-time
changes are rejected. The trainer skips committed chunks, reuses cumulative totals, and appends to
the existing metrics timeline.

To add more corpus data to the current model, run `train.py` with explicit inputs and omit
`--reset`:

```bash
PYTHONPATH=src python3 src/train.py datasets/GPTeacher.json \
  --ngram-order 3 \
  --json-chunk-size 100 \
  --max-runtime-seconds 1200 \
  --profile-ingest
```

The sibling `datasets/GPTeacher.config.json` is discovered automatically.

### Useful training options

| Option | Purpose |
| --- | --- |
| `--ngram-order 0` | Auto-select the order from a corpus sample. |
| `--ngram-order 3` | Practical low-order validation; avoids merge-token training. |
| `--ngram-order 5` or higher | Enables merge-token mechanics by default and is substantially heavier. |
| `--json-chunk-size N` | Bound JSON/NDJSON staging and define recoverable chunk boundaries. |
| `--max-json-lines N` | Limit a run to the first `N` structured records. |
| `--max-runtime-seconds N` | Pause after the active chunk once the runtime budget is reached. |
| `--chunk-eval-percent P` | Reserve `P` percent of each JSON chunk for immediate evaluation. |
| `--eval-interval N` | Run a periodic probe after crossing `N` more ingested tokens; `0` disables it. |
| `--eval-dataset PATH` | Use a separate JSON/NDJSON hold-out dataset. |
| `--metrics-export PATH` | Write the rolling evaluation and profiling timeline as JSON; `-` disables it. |
| `--profile-ingest` | Report per-chunk latency, memory, and resource deltas. |
| `--context-dimensions PRESET` | Select `default`, `deep`, `shallow`, explicit ranges, or `off`. |
| `--sentence-splitting` | Opt into punctuation-based sentence segmentation; it is off by default. |
| `--prep-workers N` | Control parallel dependency parsing and corpus staging. |
| `--graph-memory` / `--no-graph-memory` | Record (or skip) graph context memory for this run. |
| `--graph-term-index-rebuild` | Rebuild Cheetah's lexical seed index after ingest. |

Run the live parser for the complete option reference:

```bash
PYTHONPATH=src python3 src/train.py --help
```

## Graph context memory

Cheetah stores a property graph alongside the key/value tables, with associative recall
(`GRAPH_RECALL`) that spreads activation from several seeds at once. DB-SLM can use it as an extra
context signal. It requires the Cheetah backend and is **off by default** because it adds graph
writes during training.

Enable it in `.env` with `DBSLM_GRAPH_MEMORY=1`, or per run with `--graph-memory`.

**During training**, each staged prompt/response pair is recorded as:

- a node per dataset context value (`ctx:emotion:joy`), labelled `dbslm_context`;
- a node per content term (`term:kindness`), labelled `dbslm_term`, preferring dependency lemmas
  over surface words;
- `evokes` edges from context to response terms, `precedes` edges from prompt to response terms, and
  `dep_<label>` edges from the response dependency arcs;
- the complete response sentence stored as a bounded node *reference* — readable provenance that
  also feeds Cheetah's lexical seed index.

Nothing written here enters the n-gram corpus. Like the dependency layers it reads, graph memory is
a side channel.

**During inference**, the turn's dataset context values and content words become recall seeds. The
returned associations bias decoding toward recalled terms, and the hydrated reference sentences widen
the internal bias/embedding context. As with the Level 3 concept summary, that recalled text is
never shown in the response.

```bash
# train with graph memory and make free-text recall seeds resolvable
DBSLM_GRAPH_MEMORY=1 PYTHONPATH=src python3 src/train.py datasets/emotion_data.json \
  --graph-memory \
  --graph-term-index-rebuild \
  --max-json-lines 200 \
  --max-runtime-seconds 900
```

```bash
# recall it at inference and log what each turn recalled
PYTHONPATH=src python3 src/run.py \
  --graph-memory \
  --graph-recall-log \
  --prompt "Why does gratitude change how a difficult week feels?"
```

Configuration:

| Setting | CLI | Purpose |
| --- | --- | --- |
| `DBSLM_GRAPH_MEMORY` | `--graph-memory` / `--no-graph-memory` | Enable graph writes and recall. |
| `DBSLM_GRAPH_RECALL_HOPS` | — | Conceptual depth of a recall walk (1–6). |
| `DBSLM_GRAPH_RECALL_PRECISION` | — | Belief threshold an association must pass (0–1). |
| `DBSLM_GRAPH_RECALL_LIMIT` | — | Maximum associations returned per turn. |
| `DBSLM_GRAPH_RECALL_REFERENCES` | — | Maximum reference sentences hydrated per turn; `0` disables hydration. |
| `DBSLM_GRAPH_BIAS_WEIGHT` | `--graph-bias-weight` | How strongly recalled terms bias decoding; `0` keeps the context text only. |
| — | `--graph-memory-terms N` | Terms minted as nodes per record side. |
| — | `--graph-memory-dependency-arcs N` | Dependency arcs recorded as typed edges per record. |
| — | `--graph-memory-max-records N` | Records per chunk observed into the graph; `0` for all. |
| — | `--graph-recall-log` | Log seeds, association count, and truncation per turn. |

A database trained before this feature — or with `CHEETAH_GRAPH_TERM_INDEX=0` — needs
`--graph-term-index-rebuild` before free-text seeds resolve. Exact ids and declared synonym edges
work regardless.

## Run inference

Start inference only after Cheetah is running, and keep the same `.env` used for training.

### One prompt

```bash
PYTHONPATH=src python3 src/run.py \
  --prompt "How can curiosity help someone navigate an ethical dilemma?" \
  --user demo-user \
  --agent db-slm \
  --max-response-words 80
```

### Interactive conversation

```bash
PYTHONPATH=src python3 src/run.py
```

The REPL supports:

- `:history` to print the persisted Level 2 context;
- `:status` or `:conversation` to show the conversation ID and context configuration;
- `:exit` or `:quit` to close the session.

The conversation ID is printed at startup. Resume it later with:

```bash
PYTHONPATH=src python3 src/run.py \
  --conversation YOUR-CONVERSATION-ID
```

Add a stable instruction before every interactive or one-shot prompt with:

```bash
PYTHONPATH=src python3 src/run.py \
  --instruction "Answer as a concise research assistant." \
  --prompt "Summarize the role of the Cheetah hot path."
```

### Inspect Cheetah during inference

The inference CLI can print server statistics, summarize namespaces, and show prediction-table
queries:

```bash
PYTHONPATH=src python3 src/run.py \
  --cheetah-system-stats \
  --cheetah-summary "ctx:" \
  --cheetah-summary "prob:3" \
  --cheetah-predict-log \
  --prompt "What patterns are active for this question?"
```

Useful inference options:

| Option | Purpose |
| --- | --- |
| `--prompt TEXT` | Run one prompt and exit; omit it for the REPL. |
| `--conversation ID` | Continue an existing persisted conversation. |
| `--instruction TEXT` | Prefix every prompt with an instruction block. |
| `--max-turns N` | Stop an interactive session after `N` user turns. |
| `--max-response-words N` | Limit displayed response length; non-positive values disable trimming. |
| `--context-dimensions VALUE` | Override stored context dimensions for this process. |
| `--cheetah-system-stats` | Print Cheetah resource and cache statistics before prompting. |
| `--cheetah-summary PREFIX` | Summarize a Cheetah namespace; repeat for more than one. |
| `--cheetah-predict-log` | Log a prediction query after each response. |
| `--graph-memory` / `--no-graph-memory` | Recall (or skip) graph context memory before decoding. |
| `--graph-bias-weight W` | How strongly recalled graph terms bias decoding (0–1). |
| `--graph-recall-log` | Log the seeds, association count, and truncation flag of each recall. |

Run the complete inference reference with:

```bash
PYTHONPATH=src python3 src/run.py --help
```

## Runtime files

Runtime state is intentionally ignored by Git:

- the relational companion and its WAL/SHM files;
- Cheetah data under the server's configured data directory;
- `var/train_resume.json`;
- `var/eval_logs/*.json` and the quality retraining queue;
- server, training, and smoke logs;
- PID files, model caches, and the built Cheetah binary.

Metrics files are updated while training is active with `status=running`, then finalized as
`success`, `paused`, or `aborted`. Low-quality evaluation samples may also be appended to
`var/eval_logs/quality_retrain_queue.jsonl`.

There is no automated backup or migration command. Back up the Cheetah database directory and its
matching relational companion before destructive experiments.

## Troubleshooting

### Cheetah connection fails at startup

Confirm the owned session and log:

```bash
screen -ls
tmux ls
tail -n 100 var/cheetah-server.log
```

Then check that `DBSLM_CHEETAH_HOST`, `DBSLM_CHEETAH_PORT`, and
`DBSLM_CHEETAH_DATABASE` match the running server. `0.0.0.0` is a listen address and is not valid as
the Python client's destination.

### Training reports missing requirements

Verify Java and the Python environment:

```bash
java -version
python -c "import language_tool_python; print('language-tool-python available')"
```

If dependency annotations are unavailable, install a spaCy model:

```bash
python -m spacy download en_core_web_sm
```

The trainer can continue without dependency annotations, but it cannot start without a working
LanguageTool/Java installation.

### Avoid the sentence-embedding download

Force deterministic hashed embedding guidance:

```bash
DBSLM_EMBEDDER_OFFLINE=1 PYTHONPATH=src python3 src/train.py YOUR_DATASET.json
```

### A quick model produces poor output

That is expected. Small slices validate storage, framing, training, and inference; they do not
provide enough statistical coverage for strong generation. Consult the dated evidence in
[`studies/BENCHMARKS.md`](studies/BENCHMARKS.md) before comparing throughput or quality.

## Validation and development

Run the Python regression suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_paraphraser_regression.py
```

Preview the smoke matrix without changing model state:

```bash
python3 scripts/smoke_train.py --dry-run
```

The supported Cheetah-backed smoke workload is launched in its own bounded session:

```bash
scripts/start_cheetah_smoke_session.sh
```

Contributors should read [`AGENTS.md`](AGENTS.md). Generic Cheetah server changes belong in the
[`cheetah-db`](cheetah-db/) submodule and follow
[`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md).

## Further reading

- [`studies/CONCEPT.md`](studies/CONCEPT.md) — original three-level architecture
- [`studies/DB_SLM_DATABASE_AND_ALGORITHMS.md`](studies/DB_SLM_DATABASE_AND_ALGORITHMS.md) —
  algorithms and relational design
- [`studies/ALGORITHMS_FLOW.md`](studies/ALGORITHMS_FLOW.md) — compact training and inference flow
- [`studies/BENCHMARKS.md`](studies/BENCHMARKS.md) — reproducible measured runs
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — active research backlog

This repository is licensed under the [Apache License 2.0](LICENSE).
