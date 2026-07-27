# lmdb / DB-SLM — AI Agent Operational Reference

This is the fast-access operational map for `lmdb`, an experimental database-native statistical
language model (DB-SLM). The Python stack implements n-gram generation, conversational memory,
concept templates, evaluation, and Cheetah-backed prediction signals. It is research software, not a
production large language model, not the Lightning Memory-Mapped Database library, and not proof
that the planned quality or scale targets have been achieved.

## Read Order and Sources of Truth

Use the following authority order. If two sources disagree, inspect the enforcing code and focused
tests, correct stale documentation in the same task, and record unresolved contradictions under
[Known Gaps](#known-gaps).

1. [`LICENSE`](LICENSE) — legal terms for this repository.
2. [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py), the executable dataset mappings
   in [`datasets/`](datasets/), and the tests governed by
   [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md) — executable protocol and regression contracts.
3. [`src/`](src/), [`scripts/`](scripts/), [`Makefile`](Makefile), [`requirements.txt`](requirements.txt),
   and [`.env.example`](.env.example) — current implementation, command, dependency, and
   configuration behavior.
4. [`README.md`](README.md) — user-facing setup and CLI documentation. It does not override source
   behavior.
5. [`studies/DB_SLM_DATABASE_AND_ALGORITHMS.md`](studies/DB_SLM_DATABASE_AND_ALGORITHMS.md) —
   algorithm and relational-schema design. Sections that still discuss MariaDB or future work are
   design history unless current source implements them.
6. [`NEXT_STEPS.md`](NEXT_STEPS.md) — the single active backlog. A backlog item is not shipped
   behavior.
7. [`studies/BENCHMARKS.md`](studies/BENCHMARKS.md) and the remaining [`studies/`](studies/) notes —
   measured results, rationale, and historical research. Dates and performance numbers belong
   there, not in this handbook.

The root handbook governs the Python DB-SLM application and its orchestration. The nested
[`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md) is mandatory and more specific for the Go submodule,
its wire protocol, on-disk formats, tests, runtime configuration, and server operations.

## Collaboration and Maintenance Rules

- Read this file completely, check `git status`, and locate every nested `AGENTS.md` applicable to
  the files being changed before editing. Preserve unrelated user changes and untracked data.
- Keep one owner for each kind of knowledge: user workflows in [`README.md`](README.md), active work
  in [`NEXT_STEPS.md`](NEXT_STEPS.md), research and benchmark evidence in [`studies/`](studies/), and
  durable agent-facing contracts and ownership here. This file is the only root operational
  handbook; MUST NOT recreate a second reference with overlapping authority.
- Update this handbook in the same change whenever a durable fact changes: file ownership, public
  symbols, CLI or protocol surfaces, configuration, persistence, lifecycle, feature status,
  recurring failure prevention, or focused test ownership.
- Mirror user-visible commands and configuration changes into [`README.md`](README.md). Record
  studies and measurements under [`studies/`](studies/) and cross-link them; update
  [`NEXT_STEPS.md`](NEXT_STEPS.md) first for new or completed backlog work.
- Run focused checks for every changed subsystem and report checks not run. Do not claim benchmark,
  quality, or scale improvements without recording a reproducible run in
  [`studies/BENCHMARKS.md`](studies/BENCHMARKS.md).
- Treat `.env`, `var/`, raw `datasets/*.json`, Cheetah data directories, logs, PID files, SQLite
  files/WALs, model caches, and built binaries as local runtime state. They are ignored and MUST NOT
  become implementation owners or be committed.
- Keep generic database-server work in the [`cheetah-db`](cheetah-db/) submodule. Commit and test
  the change in the Cheetah repository, then advance this repository's gitlink. Keep only DB-SLM
  adapters and orchestration in this repository; never re-vendor Cheetah source.
- Launch every long-running service, smoke train, benchmark, or CI-style workload in `screen`.
  Before launch, inspect `screen -ls` for a lingering session, use an explicit timeout of at most
  30 minutes by default (at most one hour only when justified before launch), and monitor its log
  while it runs. If `screen` cannot stay alive, use `tmux` with the same checks and record the
  substitution. Prefer the bounded helpers in [`scripts/`](scripts/).
- Before a commit, review the full diff, confirm `AGENTS.md` describes the post-commit tree, validate
  local links and exact commands touched, and make sure no runtime artifacts or secrets are staged.

## Essential Project Principles

### Database-native generation is the product boundary

- Core token generation MUST flow through stored vocabulary, n-gram, cache, bias, and concept data;
  do not replace it with a transformer call while presenting the result as DB-SLM behavior.
- Sentence Transformers and the CoLA classifier are optional guidance/evaluation components. They
  may use PyTorch, CUDA, or Metal Performance Shaders (MPS), but they do not own token generation.

### Cheetah is the Level 1 operational path

- With `DBSLM_BACKEND=cheetah-db`, Level 1 contexts, counts, probabilities, Top-K rows, metadata,
  and prediction signals MUST use the Cheetah adapter. SQLite remains the relational working store
  for schema bootstrap, Level 2/3 state, and scratch/export workflows.
- Do not hide Cheetah failures by silently changing backends. A SQLite-only run must be explicitly
  requested and treated as a reduced-capability experiment.

### Dataset configuration owns corpus framing

- JSON/NDJSON field names, prompt/response labels, context placement, and canonical tags MUST come
  from the adjacent `*.config.json` or an explicit override. Dataset-specific field assumptions
  MUST NOT be hard-coded into the engine.
- Training and evaluation MUST compose the same prompt framing, including the terminal response
  label, so held-out probes test the distribution that was trained.

### Stored state and wire formats are contracts

- Context hashes, quantized values, Cheetah namespaces, serialized fixed-size payloads, and metadata
  keys are cross-process compatibility surfaces. Change them only with round-trip coverage and an
  explicit compatibility/migration decision.
- Caches are derived acceleration state. They MUST be invalidated or refreshed with the underlying
  mutation; never make a warm-cache-only behavior the correctness path.

### Expensive work must be bounded and observable

- Corpus staging, evaluation retry loops, reducer polling, queue drains, services, and benchmarks
  MUST have caps, timeouts, progress signals, and recoverable state where supported.
- Adapt concurrency to host load and preserve logs/metrics needed to explain stalls or regressions;
  do not introduce unbounded worker, socket, cache, or file-handle growth.

## Critical Implementation Contracts

- **Prompt tags are atomic and banned from generated content.**
  [`collect_prompt_tag_tokens`](src/train.py),
  [`DatasetConfig.prompt_tag_tokens`](src/db_slm/dataset_config.py), and
  [`DBSLMEngine.register_prompt_tags`](src/db_slm/pipeline.py) enumerate built-in and
  dataset-specific tags and configured plain-text frame labels before ingest/evaluation.
  `DBSLMEngine.respond` bans their token IDs,
  [`TokenScoringPipeline.score`](src/db_slm/scoring.py) preserves those bans while mixing the session
  cache and prediction distribution, and the engine performs case-normalized alias checks over each
  candidate. Plain labels retain the delimiter in their alias check, so `Emotion:` is rejected
  without banning ordinary prose containing `emotion`. Empty and marker-only candidates are also
  rejected before `|END|` stripping. The engine
  retries with fresh randomness, clears rejected token IDs, and emits a scaffold-free backstop after
  retry exhaustion even when response framing is disabled for raw evaluation. Preserve this
  sequence and the scoring/pipeline/formatter regressions.
- **Every decoder prompt ends in the selected response label.**
  [`ensure_response_prompt_tag`](src/db_slm/prompt_tags.py) is called by the training evaluation
  loader and both inference modes. [`DatasetConfig.compose_prompt`](src/db_slm/dataset_config.py)
  also emits before/after-prompt context and canonical tags identically for training and evaluation.
  Omitting the terminal `|RESPONSE|:` or its preceding dataset context makes the model continue an
  unseen frame. Staged responses end with `|END|`; user-visible output strips it through
  [`src/db_slm/text_markers.py`](src/db_slm/text_markers.py), including case-insensitive variants
  with whitespace inside the pipes that can survive from legacy fragmented counts.
- **Internal Level 3 context never becomes visible scaffold.**
  [`DBSLMEngine.respond`](src/db_slm/pipeline.py) may add `ContextSummary` text to the rolling
  context and embedding input, but it MUST NOT prepend the internal `|CONTEXT|:` payload to the
  response. [`TaggedResponseFormatter`](src/db_slm/pipeline.py) removes an already-terminal response
  frame before wrapping to prevent nested `|USER|:`/`|RESPONSE|:` blocks.
- **Dependency analysis is a side channel, not generation text.**
  JSON/NDJSON staging stores `DependencyLayer` objects on `EvaluationRecord` for alignment metrics
  and prediction training, but MUST NOT append serialized `token`/`lemma`/`head`/`dep`/`pos` fields
  to the n-gram corpus. `DBSLMEngine.respond` rejects those field-shaped artifacts so models trained
  before this repair cannot expose the internal record. Preserve the staging and response
  regressions in [`tests/test_train_monitor.py`](tests/test_train_monitor.py) and
  [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py).
- **Cheetah reducer transport has two encoding layers.**
  [`CheetahClient.pair_reduce`](src/db_slm/adapters/cheetah.py) uses canonical
  `JOB submit command=<base64>` → `JOB status id=...` → `JOB fetch id=...`, falling back to legacy
  `PAIR_REDUCE_ASYNC`/`PAIR_REDUCE_FETCH` only when the server rejects `JOB`.
  [`CheetahClient.decode_reduced_payload`](src/db_slm/adapters/cheetah.py) removes the storage
  base64 layer before [`CheetahSerializer`](src/db_slm/adapters/cheetah.py) decodes DB-SLM bytes.
  Both paths are covered in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py).
- **Cheetah pair registrations are verified and concurrency-safe.**
  [`CheetahHotPathAdapter._register_pair`](src/db_slm/adapters/cheetah.py) retries `PAIR_SET` and
  confirms with `PAIR_GET`. The submodule gitlink currently points to repair commit `6866be9`, whose
  atomic high-water mark and serialized pair mutations prevent equal-size inserts and
  shared-prefix writes from overwriting each other. Before advancing the gitlink, run the focused
  Cheetah regression tests listed in [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md).
- **Relational ingest counts refresh Cheetah before smoothing.**
  [`NGramStore.ingest`](src/db_slm/level1.py) writes the relational working counts, then
  [`MKNSmoother._collect_context_followers`](src/db_slm/level1.py) selects those current rows and
  [`MKNSmoother._mirror_counts`](src/db_slm/level1.py) publishes only changed follower sets.
  `HotPathAdapter.flush_pending` makes those async writes visible before the MKN pass continues.
  Never prefer an already-populated Cheetah count namespace over newer SQLite ingest rows; that
  freezes training at the first mirrored seed corpus.
- **Graph context memory is a side channel, never corpus text.**
  [`GraphContextMemory`](src/db_slm/graph_memory.py) writes `ctx:<field>:<value>` and `term:<lemma>`
  nodes plus `evokes`/`precedes`/`dep_<label>` edges from staged
  `EvaluationRecord`s, and hydrates them back through `GRAPH_RECALL`. Like the dependency layers it
  reads, nothing it produces may enter `CorpusChunk.train_text`, and the recalled reference sentences
  are internal bias/embedding context only — `DBSLMEngine.respond` extends `bias_context` with them
  and MUST NOT prepend them to the response. Recalled terms reach the decoder as a transient
  `extra_bias` map that [`TokenScoringPipeline.score`](src/db_slm/scoring.py) filters against the ban
  set, and [`DBSLMEngine._graph_token_bias`](src/db_slm/pipeline.py) resolves them with
  `Vocabulary.lookup` so recall can never mint vocabulary at inference time.
- **Graph ids are slugged and encoded in the adapter, not the caller.**
  Cheetah splits `GRAPH_*` arguments on whitespace, so
  [`slugify`](src/db_slm/graph_memory.py) produces single-token ids and the `_graph_*` helpers in
  [`adapters/cheetah.py`](src/db_slm/adapters/cheetah.py) base64-encode props, references, batch
  items, and any seed list containing spaces. `_graph_token` rejects a value that is not a single
  token rather than emitting a silently truncated id. Recall bounds (`hops`, `branch_limit`,
  `budget`, `reference_limit`) are clamped to the server maxima mirrored in
  [`cheetah_types.py`](src/db_slm/cheetah_types.py).
- **Node references are replaced, not merged, by `GRAPH_NODE_SET`.**
  `GraphContextMemory._write_node` reads the stored list back with `GRAPH_NODE_GET` on the first
  touch of a node in a process, then merges locally under the 64-reference server cap, so a second
  training run extends provenance instead of erasing it. Recall requests `include_seeds=1` whenever
  references are hydrated — a seed node is otherwise excluded from its own answer — and
  `signal_from_result` drops seed terms from the bias while keeping their sentences.
- **Cheetah visibility and reducer extension are server-owned.**
  `PAIR_SET_HIDDEN` stores terminals excluded from default `PAIR_SCAN`/`PAIR_REDUCE`/`PAIR_SUMMARY`;
  callers must request `include_hidden=1` to inspect cached joins. New reducer modes belong in the
  registry in [`cheetah-db/src/reducers.go`](cheetah-db/src/reducers.go), not in Python command
  dispatch. Follow the focused server tests in [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md).
- **Metadata keys have one canonical `meta:` prefix.**
  [`ConversationMemory`](src/db_slm/level2.py), [`SessionCache`](src/db_slm/level2.py), and
  [`BiasEngine`](src/db_slm/level2.py) write `meta:l2:*` values through the hot-path adapter.
  Readers accept the historical `meta:meta:l2:*` form only for compatibility; new writes MUST NOT
  recreate it.
- **Context and merge settings persist across processes.**
  [`DBSLMEngine._init_context_dimensions`](src/db_slm/pipeline.py) and
  [`DBSLMEngine._init_token_merging`](src/db_slm/pipeline.py) read Cheetah metadata first, fall back
  to SQLite metadata, and write the resolved values to both. Inference with CLI overrides may
  intentionally differ, but do not silently reinterpret an existing model.
- **`--reset` is destructive across both stores.**
  [`resolve_db_path`](src/train.py) removes the selected SQLite database plus WAL/SHM artifacts, and
  [`reset_cheetah_store`](src/train.py) attempts `RESET_DB` before progressively slower namespace
  purge fallbacks. Always select and verify the intended `DBSLM_CHEETAH_DATABASE`; never run a reset
  against an ambiguous shared namespace.
- **A configured Cheetah backend currently fails closed at startup.**
  [`build_cheetah_adapter`](src/db_slm/adapters/cheetah.py) raises `SystemExit` when the warm
  connection fails. Consequently, the later `--backonsqlite` branch in [`src/train.py`](src/train.py)
  is not reached for that failure path. Do not claim that the flag currently rescues an unreachable
  configured backend; this active contradiction is tracked under [Known Gaps](#known-gaps).
- **Long-running process ownership is explicit.**
  [`scripts/start_cheetah_server.sh`](scripts/start_cheetah_server.sh) records the exact server PID,
  refuses a live duplicate session, and attaches a timeout. Stop through
  [`scripts/stop_cheetah_server.sh`](scripts/stop_cheetah_server.sh); do not use broad process kills
  when the PID/session helpers can identify the target.

## Architecture and Data/Control Flow

Training:

`src/train.py` → settings/CLI validation → dataset config + parallel corpus/dependency staging →
`DBSLMEngine.train_from_text` → tokenizer/vocabulary → `NGramStore.ingest` →
`MKNSmoother.rebuild_all` → SQLite relational tables + Cheetah namespaces → periodic evaluation,
quality queue, adversarial prediction updates, optional graph context memory writes, metrics, and
resume state.

Inference:

`src/run.py` parent REPL → spawned `PromptWorker` → `issue_prompt` → `DBSLMEngine.respond` →
Level 3 concept signal → optional `GraphContextMemory.recall` → Level 2 history/cache/bias →
`Decoder.decode` → `TokenScoringPipeline.score` → Cheetah Top-K/prediction data with n-gram fallback →
prompt-tag guard, response backstop, formatter, and Level 2 message log.

External process boundary:

`CheetahHotPathAdapter` → thread-local `CheetahClient` TCP connections → unauthenticated Cheetah text
protocol → Cheetah pair tries, reducers, managed files, and prediction tables. The Go service and its
storage are owned by the submodule; Python owns serialization, namespace conventions, retries, and
DB-SLM orchestration.

Persistence ownership:

- SQLite owns the relational schema in [`DatabaseEnvironment`](src/db_slm/db.py), including
  vocabulary, counts/probability staging, conversation history, biases, and concepts.
- Cheetah owns its named on-disk database and serves Level 1 mirrors, metadata, reducers, and
  prediction tables. Its internal file formats are governed exclusively by
  [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md).
- `var/` owns disposable local SQLite files, logs, metrics, queue data, PID files, and resume state.
  These are runtime artifacts, not source.

## Linked Source Tree and File Reference

### [`LICENSE`](LICENSE)

Apache License 2.0 terms for the repository.

- **Boundary:** legal terms outrank all implementation and documentation guidance.

### [`README.md`](README.md)

User-facing overview, environment setup, trainer/inference argument guide, Cheetah operations, and
smoke workflows. Update it for user-visible behavior; do not place internal ownership rules here.

- **Depends on:** CLI parsers in [`src/train.py`](src/train.py) and [`src/run.py`](src/run.py),
  [`.env.example`](.env.example), and helper scripts.
- **Common mistake:** examples can become stale as parser defaults change; verify them against
  `--help` and source before copying them into this handbook.

### [`NEXT_STEPS.md`](NEXT_STEPS.md)

Single active backlog plus a compact completed-work context. New defects, experiments, and
optimizations belong here before they are summarized elsewhere.

- **Current direction:** scale the repaired Cheetah-only validation, evaluate deep prediction
  layers, diagnose punctuation collapse, and repair the ineffective unreachable-backend fallback.
- **Common mistake:** completed bullets are historical evidence, not a replacement for feature
  status or regression tests.

### [`Makefile`](Makefile)

Owns the `smoke-train` wrapper and destructive `clean-smoke` cleanup target.

- **Key targets:** `smoke-train` forwards `SMOKE_SCENARIOS`, `SMOKE_DATASET`, `SMOKE_MATRIX`, and
  `SMOKE_BENCH`; `clean-smoke` removes named SQLite files and `var/smoke_train`.
- **Depends on:** [`scripts/smoke_train.py`](scripts/smoke_train.py).
- **Common mistake:** `make smoke-train` is long-running and mutates runtime databases; launch it
  through a bounded screen/tmux session.

### [`requirements.txt`](requirements.txt)

Declares runtime Python dependencies for embeddings, grammar scoring, tokenizers, dependency
parsing, and resource metrics.

- **Important boundary:** PyTorch and Transformers arrive transitively; source handles optional
  accelerator/model availability, but `src/train.py` treats `language_tool_python` plus Java as
  mandatory startup requirements.
- **Common mistake:** installing this file does not install a spaCy language model or Java.

### [`.env.example`](.env.example)

Tracked configuration template for backend selection, SQLite scratch path, Cheetah connection and
idle grace, reducer polling, dataset, embedder, sentence splitting, graph context memory, and SQLite
flush thresholds.

- **Loaded by:** [`load_settings`](src/db_slm/settings.py) reads `.env`, then lets real environment
  variables override it. It also carries the graph context memory defaults (`DBSLM_GRAPH_MEMORY`,
  `DBSLM_GRAPH_RECALL_*`, `DBSLM_GRAPH_BIAS_WEIGHT`).
- **Common mistake:** copy to the ignored `.env`; never add real host credentials or local paths to
  this template.

### [`.gitmodules`](.gitmodules)

Pins `cheetah-db/` to `https://github.com/cekkr/cheetah.git`.

- **Change rule:** generic Cheetah changes are committed in the submodule first; the parent change
  is only the resulting gitlink.

### [`.gitignore`](.gitignore)

Defines generated/runtime boundaries: Python caches and environments, `.env`, `var/`, raw dataset
payloads except configs, Cheetah data, and the built server binary.

- **Common mistake:** ignored state can still be essential to a local run; never clean it
  destructively merely to obtain a tidy tree.

### [`.vscode/settings.json`](.vscode/settings.json)

Workspace-only editor defaults for the system Python environment and enlarged console/terminal
scrollback.

- **Boundary:** this file does not define runtime dependencies, a supported interpreter, or a test
  command.

### [`datasets/GPTeacher.config.json`](datasets/GPTeacher.config.json)

Maps `input` to `|USER|`, `response` to `|RESPONSE|`, and prepends the `instruction` field under
`|INSTRUCTION|`.

- **Loaded by:** [`load_dataset_config`](src/db_slm/dataset_config.py) through adjacent filename
  inference.
- **Common mistake:** the raw `datasets/GPTeacher.json` corpus is intentionally untracked; do not
  make the config depend on an absolute local path.

### [`datasets/emotion_data.config.json`](datasets/emotion_data.config.json)

Maps the emotion corpus's `prompt`/`response` fields and converts `emotion` into the opt-in
canonical `|CTX|` context header.

- **Loaded by:** training/evaluation corpus staging in [`src/train.py`](src/train.py).
- **Common mistake:** `|CTX|` is emitted only because this config declares `canonical_tag`; the
  engine MUST NOT add it to unrelated datasets.

### [`datasets.md`](datasets.md)

Records dataset provenance, shape, and practical length statistics used to choose chunk/evaluation
limits.

- **Authority:** descriptive dataset evidence only; executable mappings remain in the adjacent
  configs.

### [`src/train.py`](src/train.py)

Canonical training CLI and the largest orchestration boundary. It owns argument parsing,
requirements checks, reset/resume, auto n-gram selection, corpus staging, dependency annotations,
ingest/evaluation scheduling, prediction-table training, adversarial updates, metrics, and shutdown.

- **Key functions and subparts:** `build_parser` defines the public CLI; `reset_cheetah_store`
  coordinates destructive reset; `collect_prompt_tag_tokens`, `iter_json_chunks`, and
  `parallel_corpus_stream` stage data; `AdversarialTrainer`, `DecoderPenaltyTuner`,
  `InferenceMonitor`, and `IngestProfiler` own post-ingest feedback and observability;
  `_configure_graph_memory`, `_observe_graph_memory`, and `_rebuild_graph_term_index` own the
  optional graph context memory pass; `main` fixes ordering across all phases.
- **Depends on:** nearly every [`src/db_slm/`](src/db_slm/) subsystem plus dataset configs and
  [`src/helpers/resource_monitor.py`](src/helpers/resource_monitor.py).
- **Tests:** indirect adapter coverage in
  [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py); no focused automated coverage
  for resume, reset, chunking, or penalty tuning.
- **Common mistakes:** register every dataset/evaluation tag before probes; keep stdin synchronous;
  verbose staging lines are decoder-input/reference previews rather than generated responses; never
  append serialized dependency records to generation text; never replay every crossed
  `--eval-interval` after a large chunk; never run `--reset` without an isolated Cheetah database;
  escape literal `%` as `%%` in argparse help strings or `--help` crashes during interpolation.
- **Bounded resume:** `--max-runtime-seconds` finishes the active chunk, records `status=paused`, and
  exits at a recoverable boundary; invoking `train.py` with no arguments resumes the saved command
  and cumulative totals.

### [`src/run.py`](src/run.py)

Inference CLI and REPL. The parent process owns terminal I/O while `PromptWorker` starts a spawned
decoder process and proxies prompt, history, status, and shutdown messages.

- **Key functions and subparts:** `build_parser`; `build_prompt_formatter`; `PromptWorker`;
  `_decoder_worker`; `respond_once_worker`; `interactive_loop`; `build_response_formatter`.
- **Graph options:** `--graph-memory`/`--no-graph-memory`, `--graph-bias-weight`, and
  `--graph-recall-log` travel through `PromptWorker` into the spawned decoder process; every new
  worker option must be added to both the constructor and the `_decoder_worker` positional list.
- **Depends on:** [`issue_prompt`](src/db_slm/inference_shared.py), Cheetah inspection helpers,
  prompt-tag normalization, and persisted SQLite conversation state.
- **Tests:** no focused CLI/worker lifecycle test.
- **Common mistakes:** preserve the spawn-safe module entry point; always call
  `PromptWorker.close`; append the response tag after formatting and before dispatch.

### [`src/log_helpers.py`](src/log_helpers.py)

Central elapsed-time log prefix and verbosity gate used by trainer, inference, evaluation, and
helpers.

- **Key functions:** `log` emits `+[seconds]`; `log_verbose` gates `LMDB_LOG_LEVEL`; `reset_timestamp`
  resets the process epoch.
- **Common mistake:** bypassing this helper makes long-run logs impossible to correlate.

### [`src/__init__.py`](src/__init__.py)

Empty package marker for the top-level source tree.

- **Boundary:** it exports no public behavior; the supported package façade is
  [`src/db_slm/__init__.py`](src/db_slm/__init__.py).

### [`src/db_slm/__init__.py`](src/db_slm/__init__.py)

Public package façade re-exporting `DatabaseEnvironment` and `DBSLMEngine`.

- **Depends on:** [`src/db_slm/db.py`](src/db_slm/db.py) and
  [`src/db_slm/pipeline.py`](src/db_slm/pipeline.py).
- **Common mistake:** do not grow this into a second orchestration layer.

### [`src/db_slm/settings.py`](src/db_slm/settings.py)

Owns `.env` parsing and the immutable `DBSLMSettings` configuration object.

- **Key functions:** `_parse_env_file`; `load_settings`; `DBSLMSettings.sqlite_dsn`.
- **Resolution order:** real environment → `.env` → code default.
- **Common mistake:** new settings require updates here, [`.env.example`](.env.example),
  [`README.md`](README.md), and this handbook.

### [`src/db_slm/db.py`](src/db_slm/db.py)

Owns SQLite connection lifecycle, schema bootstrap, dynamic n-gram order tables, metadata, query
helpers, and transactions. It does not own Cheetah formats or protocol behavior.

- **Key functions and subparts:** `DatabaseEnvironment._bootstrap_schema`; `ensure_order_tables`;
  `_ensure_column`; `transaction`; `set_metadata`/`get_metadata`; `hash_tokens`.
- **Called by:** all three DB-SLM levels, pipeline startup, trainer, and inference.
- **Tests:** no dedicated schema/migration/transaction test.
- **Common mistakes:** schema additions must remain idempotent for existing files; table names
  derived from n-gram order must never contain unvalidated user text.

### [`src/db_slm/hashing.py`](src/db_slm/hashing.py)

Defines the shared deterministic hash of token ID sequences.

- **Key function:** `hash_tokens` packs unsigned 32-bit IDs and returns a fixed digest.
- **Called by:** SQLite context registry and Cheetah namespace mapping.
- **Common mistake:** a hash encoding change invalidates both stored contexts and hot-path aliases.

### [`src/db_slm/level1.py`](src/db_slm/level1.py)

Owns tokenization, vocabulary, merge-token mechanics, quantization, n-gram ingest/read paths, and
Modified Kneser–Ney (MKN) materialization.

- **Key classes:** `RegexTokenizerBackend` and `HuggingFaceTokenizerBackend`; `Vocabulary`;
  `Tokenizer` and `MergeStats`; `LogProbQuantizer`; `NGramStore`; `MKNSmoother`.
- **Called by:** [`DBSLMEngine`](src/db_slm/pipeline.py), decoder, evaluation perplexity, and
  prediction inheritance.
- **Tests:** [`tests/test_level1_smoothing.py`](tests/test_level1_smoothing.py) covers current
  relational count selection plus delta-only Cheetah mirroring; tokenizer, merge math, MKN numeric
  output, and quantization remain uncovered.
- **Common mistakes:** preserve registered structural tokens as atomic units; merging is disabled
  below n-gram order 5; publish context/count/probability/Top-K changes through `HotPathAdapter`;
  flush asynchronous count writes before a reducer can observe the namespace.

### [`src/db_slm/level2.py`](src/db_slm/level2.py)

Owns conversation messages and correction logs, the pointer-style session distribution, decode
profiles, and contextual token bias.

- **Key classes:** `ConversationMemory`; `SessionCache`; `BiasEngine`; `Message`; `Correction`.
- **Persistence:** canonical SQLite rows plus Cheetah `meta:l2:*` mirrors and compatibility reads
  for old double-prefixed metadata.
- **Tests:** no focused conversation, correction, cache, expiration, or metadata restart tests.
- **Common mistakes:** metadata mirrors accelerate cold start but do not authorize dropping the
  relational message/correction records.

### [`src/db_slm/level3.py`](src/db_slm/level3.py)

Owns concept definitions, quantized concept probabilities, templates, transient conversation
signals, payload providers, and verbalization.

- **Key classes:** `ConceptRepository`; `Verbalizer`; `ConceptPredictor`; `ConceptEngine`;
  `ConceptExecution`.
- **Called by:** [`DBSLMEngine.respond`](src/db_slm/pipeline.py) before Level 1 decoding.
- **Tests:** no focused concept-signal expiry/consumption or template test.
- **Common mistakes:** rendered `ContextSummary` is internal bias/context input, not user-visible
  response scaffolding.

### [`src/db_slm/graph_memory.py`](src/db_slm/graph_memory.py)

Owns the DB-SLM conventions layered on Cheetah's graph store: id minting, corpus-to-graph
observation, recall seed composition, and the projection of a recall answer onto decoder inputs.

- **Key symbols:** `slugify`; `context_node_id`; `term_node_id`; `term_text_from_node_id`;
  `strip_structural_tags`; `content_terms`; `dependency_terms`; `GraphIngestStats`;
  `GraphContextSignal`; `GraphContextMemory.observe_records`/`build_seeds`/`recall`/
  `signal_from_result`.
- **Conventions:** `ctx:<field>:<value>` and `term:<lemma>` ids; `dbslm_context`/`dbslm_term` labels;
  `evokes`/`precedes`/`dep_<label>` edge types; the response sentence as a bounded node reference.
- **Depends on:** the `graph_*` surface of [`HotPathAdapter`](src/db_slm/adapters/base.py) only. It
  deliberately does not import [`evaluation.py`](src/db_slm/evaluation.py); records are duck-typed to
  keep the dependency one-way.
- **Tests:** [`tests/test_graph_memory.py`](tests/test_graph_memory.py).
- **Common mistakes:** graph output is never corpus or response text; ids must stay single protocol
  tokens; per-record term/arc caps and the recall limit/branch/budget bounds exist to keep both the
  command stream and the payload finite.

### [`src/db_slm/pipeline.py`](src/db_slm/pipeline.py)

High-level façade wiring settings, SQLite, Cheetah, all three levels, embeddings, training,
corrections, and guarded response generation.

- **Key classes:** `DBSLMEngine`; `TokenMergeTracker`; `LowResourceHelper`; `SimpleParaphraser`;
  `ResponseBackstop`; `TaggedResponseFormatter`.
- **Key paths:** `train_from_text`; `register_prompt_tags`; `respond`; `context_relativism`;
  `record_correction`; `observe_graph_records`; `consume_graph_signal`; `_run_graph_layer`;
  `_graph_token_bias`.
- **Tests:** response framing in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) and
  marker-only/dependency-artifact raw-response fallback in
  [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py); paraphraser behavior in
  [`scripts/run_paraphraser_regression.py`](scripts/run_paraphraser_regression.py).
- **Common mistakes:** constructor order matters because metadata, tokenizer special tokens,
  concepts, low-resource state, and prompt-tag bans depend on earlier components; validate visible
  content before accepting a decode because `strip_end_marker("|END|")` is empty.

### [`src/db_slm/decoder.py`](src/db_slm/decoder.py)

Owns autoregressive candidate lookup/backoff, scoring delegation, top-p sampling, context-dimension
tracking, prediction-table blending, EOS handling, and optional scoring observations.

- **Key symbols:** `DecoderConfig`; `Decoder.decode`; `_resolve_candidates`; `_sample`;
  `_prediction_distribution`; `_relativistic_fallback`.
- **Depends on:** [`NGramStore`](src/db_slm/level1.py),
  [`TokenScoringPipeline`](src/db_slm/scoring.py), Level 2 cache/bias, and the hot-path adapter.
- **Tests:** no deterministic decoder/backoff/ban test.
- **Common mistakes:** prompt bans must enter before scoring; only commit accepted generation tokens
  to the session cache.

### [`src/db_slm/scoring.py`](src/db_slm/scoring.py)

Owns the composable candidate scoring order and optional per-step trace objects.

- **Key symbols:** `TokenScoringPipeline.score`; `CandidateScore`; `ScoreResult`; `ScoreSnapshot`;
  `ScoreObserver`.
- **Order:** dequantized base log probability → temperature → Level 2 bias (plus the transient
  `extra_bias` graph recall map) → repeat/dimension penalties → cache mixture → normalization →
  optional prediction blend → trace.
- **Tests:** [`tests/test_scoring.py`](tests/test_scoring.py) covers prompt-tag bans across cache and
  prediction mixing; numeric scoring order remains uncovered.
- **Common mistake:** moving normalization or mixing steps changes semantics even when the same
  inputs remain present; every source added to the distribution must preserve the decoder ban set.

### [`src/db_slm/context_dimensions.py`](src/db_slm/context_dimensions.py)

Defines context span presets, CLI parsing/serialization, and grouped repetition penalties.

- **Key symbols:** `ContextDimension`; `ContextDimensionTracker`; `parse_context_dimensions_arg`;
  `DEFAULT_CONTEXT_DIMENSIONS`; `DEEP_CONTEXT_DIMENSIONS`.
- **Called by:** both CLIs, decoder, pipeline metadata, and context-window embeddings.
- **Common mistake:** a bare sequence such as `4,8,4` means contiguous span lengths, while `1-2,3-5`
  means explicit ranges.

### [`src/db_slm/context_window_embeddings.py`](src/db_slm/context_window_embeddings.py)

Owns tag-aware window extraction, running per-dimension embedding prototypes, adaptive sampling,
similarity weights, fused context-matrix layers, and SQLite/Cheetah metadata persistence.

- **Key classes:** `ContextWindowExtractor`; `ContextDimensionPrototype`;
  `ContextWindowEmbeddingManager`.
- **Key paths:** `observe_corpus`; `weights_for_text`; `context_matrix_payload_for_text`; `flush`;
  `set_tag_enumerator`.
- **Tests:** no focused extraction, prototype persistence, tag-weight, or fusion-depth test.
- **Common mistakes:** keep the same embedder/model semantics between observe and inference; a
  dimension tag index must come from `DBSLMEngine.register_prompt_tags`.

### [`src/db_slm/sentence_parts.py`](src/db_slm/sentence_parts.py)

Owns optional punctuation segmentation, external-or-hashed embeddings, contextual header lifting,
and real-time split profiling before tokenization.

- **Key classes:** `RealtimeTokenizerProfiler`; `SentenceSegmenter`; `ExternalEmbedder`;
  `SentencePartEmbeddingPipeline`.
- **Configuration:** `DBSLM_SENTENCE_SPLIT` is off by default;
  `DBSLM_EMBEDDER_MODEL` selects a local/Hugging Face model;
  `DBSLM_EMBEDDER_OFFLINE=1` forces hashed vectors.
- **Common mistakes:** do not reintroduce dataset-specific word lists or unmanaged structural tags;
  sentence splitting is opt-in because punctuation segmentation changed generation quality.

### [`src/db_slm/dataset_config.py`](src/db_slm/dataset_config.py)

Owns executable JSON/NDJSON schema mapping and prompt composition.

- **Key classes/functions:** `DatasetConfig`; `DatasetFieldConfig`; `ContextFieldConfig`;
  `load_dataset_config`; `infer_config_path`; `_normalize_context_placement`.
- **Resolution order:** explicit override → `DBSLM_DATASET_CONFIG_PATH` → adjacent config → generic
  `prompt`/`response` defaults.
- **Tests:** [`tests/test_dataset_config.py`](tests/test_dataset_config.py) covers preface and trailing
  context/canonical-tag prompt composition.
- **Common mistake:** parsing errors currently fall through to another candidate/default; validate
  configs directly when changing schema.

### [`src/db_slm/prompt_tags.py`](src/db_slm/prompt_tags.py)

Owns the terminal response-label invariant.

- **Key symbol:** `ensure_response_prompt_tag` normalizes a label to `|TAG|:` form, avoids duplicate
  terminal tags, and appends it when absent.
- **Called by:** trainer evaluation paths and both inference modes.

### [`src/db_slm/text_markers.py`](src/db_slm/text_markers.py)

Owns the `|END|` training marker and extraction of complete user-visible response text.

- **Key symbols:** `append_end_marker`; `strip_end_marker`; `extract_complete_sentence`.
- **Common mistake:** the marker is corpus/control data and MUST NOT be returned to the user;
  stripping must recognize legacy spaced/case variants while staging continues to emit canonical
  `|END|`.

### [`src/db_slm/inference_shared.py`](src/db_slm/inference_shared.py)

Defines `issue_prompt`, the shared bridge used by training probes and the REPL to start/reuse a
conversation and call `DBSLMEngine.respond`.

- **Important options:** seeded vs seedless conversations, minimum response words, decoder config,
  RNG seed, response scaffolding, score observer, and dataset context tokens used as graph recall
  seeds.
- **Common mistake:** evaluation uses `seed_history=False`; interactive low-resource sessions may
  seed caretaker turns.

### [`src/db_slm/evaluation.py`](src/db_slm/evaluation.py)

Owns dependency-layer extraction, inference variants/retries, lexical/structural/dependency and
perplexity metrics, repetition memory, prediction probes, metrics JSON, and quality gating.

- **Key symbols:** `EvaluationRecord`; `DependencyLayer`; `build_dependency_layer`;
  `VariantSeedPlanner`; `ResponseEvaluator`; `run_inference_records`; `EvalLogWriter`;
  `QualityGate`; `ContextProbabilityProbe`.
- **Persistence:** `EvalLogWriter` atomically rewrites a `status=running` snapshot after every
  evaluation/profile event, appends existing events during resume, then finalizes the same file with
  terminal status and totals.
- **Depends on:** [`src/db_slm/quality.py`](src/db_slm/quality.py),
  [`src/db_slm/metrics.py`](src/db_slm/metrics.py), and
  [`src/helpers/char_tree_similarity.py`](src/helpers/char_tree_similarity.py).
- **Tests:** no focused retry-budget, metric, log-schema, or queue test.
- **Common mistakes:** prompt-tag retries share a bounded attempt budget; CPU-heavy quality scoring
  is load-gated; missing dependency parsers degrade with a warning.

### [`src/db_slm/quality.py`](src/db_slm/quality.py)

Owns grammar, CoLA acceptability, embedding similarity/novelty, and combined quality scoring.

- **Key classes:** `SentenceQualityScorer`; lazy `_LanguageToolProxy`; lazy `_CoLAClassifier`.
- **Resource boundary:** heavy tools load only when `AdaptiveLoadController` permits; requested
  `DEVICE=cuda|mps` falls back when unavailable.
- **Common mistake:** a missing optional CoLA model produces `None`, while training startup still
  requires LanguageTool and Java.

### [`src/db_slm/metrics.py`](src/db_slm/metrics.py)

Owns lightweight text metrics used throughout evaluation and response shaping.

- **Key functions:** `lexical_overlap`; `rouge_l_score`; `keyword_summary`.
- **Common mistake:** these surface metrics do not prove semantic correctness.

### [`src/db_slm/system.py`](src/db_slm/system.py)

Owns host-load estimation, adaptive worker suggestions, and the guard for heavy evaluation tasks.

- **Key symbols:** `headroom_ratio`; `suggest_worker_count`; `AdaptiveLoadController`.
- **Common mistake:** load average may be unavailable and uses a fallback; it is advisory, not a
  hard resource quota.

### [`src/db_slm/cheetah_types.py`](src/db_slm/cheetah_types.py)

Defines immutable Python projections for Cheetah reducers, namespace summaries, system statistics,
prediction queries, and adaptive reducer page sizing.

- **Key symbols:** `Raw*Projection`; `PredictionQueryResult`; `NamespaceSummary`;
  `CheetahSystemStats.derive_reduce_page_limit`; the `Graph*` projections
  (`GraphRecallResult`, `GraphAssociation`, `GraphReferenceSentence`, `GraphSimilarResult`,
  `GraphNodeRecord`, `GraphEdgeBatchResult`, `GraphTermIndexStats`) and the `GRAPH_*` bound
  constants mirrored from the server.
- **Common mistake:** parser/serializer changes must update these projections together.

### [`src/db_slm/cheetah_vectors.py`](src/db_slm/cheetah_vectors.py)

Defines `AbsoluteVectorOrder`, the canonical sorted byte encoding for nested token evidence used as
Cheetah trie prefixes.

- **Contracts:** version/type bytes, big-endian integers, sorted child payloads, 255-child and
  64-KiB child limits.
- **Tests:** no focused Python vector fixture; Cheetah-side semantics are documented in
  [`cheetah-db/CONCEPTS.md`](cheetah-db/CONCEPTS.md).
- **Common mistake:** changing ordering or depth truncation breaks existing aliases.

### [`src/db_slm/adapters/base.py`](src/db_slm/adapters/base.py)

Defines the complete `HotPathAdapter` protocol and `NullHotPathAdapter` SQLite-only implementation.

- **Surface:** publish/fetch Level 1 state, metadata, scans/reducers, context relativism, summaries,
  system stats, pending-write flush, prediction set/train/query/inherit operations, and the graph
  context memory calls `graph_node_set`/`graph_node_get`/`graph_edge_set_batch`/`graph_recall`/
  `graph_similar`/`graph_term_index`.
- **Called by:** Level 1, Level 2 metadata, context windows, decoder, trainer, and CLI diagnostics.
- **Common mistake:** add a new capability to the protocol, null adapter, concrete adapter, and
  callers in one change.

### [`src/db_slm/adapters/__init__.py`](src/db_slm/adapters/__init__.py)

Re-exports `HotPathAdapter` and `NullHotPathAdapter` as the adapter package's stable lightweight
surface.

- **Boundary:** the concrete Cheetah implementation is imported directly from
  [`src/db_slm/adapters/cheetah.py`](src/db_slm/adapters/cheetah.py).

### [`src/db_slm/adapters/cheetah.py`](src/db_slm/adapters/cheetah.py)

Owns the Cheetah TCP client, text protocol parsing, DB-SLM fixed payload serialization,
thread-local client pool, namespace/cache conventions, async publication, reducer pagination, and
prediction commands.

- **Key classes:** `CheetahClient`; `CheetahSerializer`; `_ThreadLocalCheetahClientPool`;
  `CheetahHotPathAdapter`; `CheetahError`; `CheetahFatalError`.
- **Key paths:** `pair_scan`; `pair_reduce`; `decode_reduced_payload`; `predict_*`; `graph_*`;
  `_register_pair`; `_edit_or_reinsert`; `build_cheetah_adapter`.
- **Graph encoding helpers:** `_graph_token`; `_graph_encode_json`; `_graph_encode_seeds`;
  `_graph_format_precision`; `_graph_association_from_payload`; `_parse_graph_recall_response`.
- **Tests:** [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) covers serialization,
  idempotent publish/fetch, response parsing, timeout recovery, canonical job flow, legacy fallback,
  and storage transport decoding.
- **Common mistakes:** `0.0.0.0` is a listen address, not a client destination; namespace values are
  bytes; do not share one socket across worker threads; flush count mirrors before reducer reads;
  fatal adapter disable must remain visible.

### [`src/helpers/char_tree_similarity.py`](src/helpers/char_tree_similarity.py)

Owns character-tree substring repetition, edit-distance, token sequence, and composite similarity
metrics used by evaluation.

- **Key symbols:** `CharTree`; `substring_multiset_similarity`; `levenshtein_char`;
  `token_sequence_similarity`; `similarity_score`.
- **Tests:** no focused metric fixtures.
- **Common mistake:** this module measures repetition/similarity, not semantic equivalence.

### [`src/helpers/cheetah_cli.py`](src/helpers/cheetah_cli.py)

Formats namespace summaries, system statistics, and prediction query results for trainer and
inference diagnostics.

- **Key functions:** `parse_summary_prefix`; `collect_namespace_summary_lines`;
  `collect_system_stats_lines`; `format_prediction_query`.
- **Common mistake:** keep parsing/formatting separate from protocol execution in the adapter.

### [`src/helpers/resource_monitor.py`](src/helpers/resource_monitor.py)

Owns portable resource snapshots and deltas for CPU, RSS, threads, disk I/O, and load.

- **Key symbols:** `ResourceSample`; `ResourceDelta`; `ResourceMonitor.snapshot`;
  `ResourceMonitor.delta`; `ResourceMonitor.to_event`.
- **Called by:** ingest profiling and metrics export.
- **Common mistake:** unsupported platform fields are optional; do not treat missing data as zero
  usage.

### [`src/helpers/torch_device.py`](src/helpers/torch_device.py)

Centralizes `DEVICE` normalization, availability checks, and automatic CUDA/MPS/CPU choice for
optional tensor-backed helpers.

- **Key functions:** `requested_device`; `device_available`; `auto_device`.
- **Common mistake:** the core DB-SLM decoder does not use this module.

### [`src/helpers/__init__.py`](src/helpers/__init__.py)

Empty package marker for operational helper modules.

- **Boundary:** import concrete helpers from their owning modules; this file exports no façade.

### [`scripts/start_cheetah_server.sh`](scripts/start_cheetah_server.sh)

Starts the built Cheetah server headlessly in a bounded `screen` session, falling back to `tmux`,
while recording an exact PID and log path.

- **Configuration:** `CHEETAH_SERVER_BIN`, `CHEETAH_SERVER_SESSION`, `CHEETAH_SERVER_LOG`,
  `CHEETAH_SERVER_PID_FILE`, `CHEETAH_SERVER_TIMEOUT`.
- **Common mistake:** it refuses a live duplicate; stop the existing owned session instead of
  overwriting it.

### [`scripts/stop_cheetah_server.sh`](scripts/stop_cheetah_server.sh)

Stops only the recorded server PID and matching screen/tmux session, escalating to `KILL` after a
short graceful wait.

- **Common mistake:** a custom start session/PID path requires the same variables at stop time.

### [`scripts/run_cheetah_smoke.sh`](scripts/run_cheetah_smoke.sh)

Runs a bounded Cheetah-backed emotion-corpus train with an isolated named Cheetah database, scratch
SQLite path, log, and metrics file.

- **Configuration:** `CHEETAH_SMOKE_*` variables define row/chunk/evaluation limits, timeout,
  database, and artifact paths.
- **Mutation:** deletes the selected scratch SQLite path and writes Cheetah/runtime state.
- **Common mistake:** this script is the workload; launch it through
  [`scripts/start_cheetah_smoke_session.sh`](scripts/start_cheetah_smoke_session.sh).

### [`scripts/start_cheetah_smoke_session.sh`](scripts/start_cheetah_smoke_session.sh)

Wraps the Cheetah smoke workload in a duplicate-safe `screen`/`tmux` session and supplies unique
artifact paths and an explicit timeout.

- **Common mistake:** it assumes the Cheetah server is already running and healthy.

### [`scripts/smoke_train.py`](scripts/smoke_train.py)

Orchestrates sequential training scenarios, tails progress, records active child PIDs and benchmark
JSON, and optionally triggers queue drains.

- **Key classes/functions:** `RunTracker`; `TelemetryMonitor`; `QueueDrainAutomation`;
  `BenchmarkRecorder`; `Scenario`; `load_scenarios`; `run_subprocess`; `main`.
- **Artifacts:** untracked `var/smoke_train/benchmarks.json`, per-scenario metrics/SQLite files, and
  active-run tracking.
- **Common mistake:** `--dry-run` still updates planning/benchmark state; real scenarios are
  long-running and belong in a bounded session.

### [`scripts/drain_queue.py`](scripts/drain_queue.py)

Owns thresholded retraining of the quality queue, command construction, metrics parsing, and
post-success queue capping.

- **Key functions:** `build_command`; `_cap_queue`; `main`.
- **Mutation:** invokes training and rewrites the queue after success.
- **Common mistake:** use `--dry-run` to inspect the command; do not hand-trim the live queue while a
  drain is running.

### [`scripts/run_paraphraser_regression.py`](scripts/run_paraphraser_regression.py)

Runs data-driven guard/rewrite cases against `SimpleParaphraser`.

- **Key functions:** `load_cases`; `run_case`; `main`.
- **Fixture:** [`studies/paraphraser_regression.jsonl`](studies/paraphraser_regression.jsonl).

### [`run.sh`](run.sh)

Legacy generic helper that starts a Go command and a Python command in detached `screen` sessions
and waits indefinitely.

- **Status:** not the supported Cheetah lifecycle path because it has no workload timeout, assumes
  `go run ./src`, and relies on its parent shell trap for cleanup.
- **Safe alternative:** use the bounded server/smoke helpers above.

### [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py)

Primary Python unit regression file.

- **Test groups:** `CheetahSerializerTests`; `CheetahHotPathAdapterTests`;
  `CheetahClientParsingTests`; `TaggedResponseFormatterTests`.
- **Depends on:** fake in-memory client only; no live Cheetah server is required.
- **Common mistake:** protocol parser changes need both success-response fixtures and transport
  decoding coverage.

### [`tests/__init__.py`](tests/__init__.py)

Package marker enabling module-style `unittest` invocation.

- **Boundary:** it contains no fixtures or test registration.

### [`tests/test_dataset_config.py`](tests/test_dataset_config.py)

Focused prompt-composition regressions using the executable GPTeacher and emotion mappings.

- **Contracts:** preface fields stay before `|USER|`; default trailing fields and canonical `|CTX|`
  tokens stay after it so training and evaluation end on the same context.

### [`tests/test_scoring.py`](tests/test_scoring.py)

Focused scoring-pipeline regressions with lightweight cache/prediction fixtures.

- **Contracts:** a banned structural token cannot re-enter the distribution or score trace through
  Level 2 cache mixture, prediction blending, or the transient graph `extra_bias` map; a graph bias
  raises an allowed candidate's log probability.

### [`tests/test_graph_memory.py`](tests/test_graph_memory.py)

Focused graph context memory regressions using a scripted client transport and a recording adapter;
no live Cheetah server is required.

- **Test groups:** `GraphIdentityTests` (id slugging and structural-tag stripping);
  `CheetahGraphProtocolTests` (argument encoding, clamped bounds, payload parsing, error and
  disabled-adapter handling); `GraphContextMemoryIngestTests` (node/edge shape, reference merging,
  per-record caps); `GraphContextMemoryRecallTests` (seed composition, `include_seeds`, term/sentence
  projection).
- **Contracts:** ids stay single protocol tokens; props/references/items/spaced seeds travel
  base64-encoded; requested bounds never exceed the server maxima; stored references are merged
  rather than overwritten; seed nodes contribute sentences but never bias.

### [`tests/test_evaluation.py`](tests/test_evaluation.py)

Focused evaluation-log writer regression.

- **Contract:** structured samples are retained in the metrics event without printing a second raw
  `sample.prompt`/`sample.generated` pair that can be mistaken for the timestamped probe result; the
  event is atomically visible in a running metrics snapshot before finalization.

### [`tests/test_train_monitor.py`](tests/test_train_monitor.py)

Focused periodic-evaluation scheduling regression.

- **Contract:** a completed chunk crossing multiple token intervals triggers one current probe and
  advances to the next future threshold instead of replaying the entire missed-threshold backlog;
  the recoverable runtime-budget CLI remains parser-visible; dependency objects remain attached to
  prediction/evaluation records without entering `CorpusChunk.train_text`.

### [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py)

Focused raw-response fallback regression using lightweight engine fixtures.

- **Contract:** a marker-only decode is retried to the configured budget and then becomes visible,
  scaffold-free backstop text instead of an empty evaluation response; serialized dependency-field
  output is rejected through the same path.

### [`tests/test_level1_smoothing.py`](tests/test_level1_smoothing.py)

Focused count-source and mirror regression using SQLite-like rows plus stale hot-path fixtures.

- **Contract:** current relational ingest counts win over stale Cheetah rows; only changed/new
  follower sets are published, and queued writes are flushed before smoothing proceeds.

### [`studies/paraphraser_regression.jsonl`](studies/paraphraser_regression.jsonl)

Data-driven cases distinguishing guarded structural/corrective/multi-turn prompts from ordinary
prompts that may be paraphrased.

- **Owner:** [`scripts/run_paraphraser_regression.py`](scripts/run_paraphraser_regression.py).

### [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md)

Mandatory scoped handbook for the Go submodule. It owns Cheetah principles, command registry,
source map, formats, tests, configuration, security, and release limitations.

- **Scope rule:** read it before running the service, touching the submodule, changing Cheetah
  namespaces/protocol assumptions, or editing `DBSLM_CHEETAH_*` interoperability.

### [`cheetah-db/CONCEPTS.md`](cheetah-db/CONCEPTS.md)

Defines Cheetah reducer and context-relativism concepts consumed by the Python adapter.

- **Boundary:** source and Cheetah tests override prose if they diverge.

### [`studies/DB_SLM_DATABASE_AND_ALGORITHMS.md`](studies/DB_SLM_DATABASE_AND_ALGORITHMS.md)

Deep design reference for tokenization, relational tables, MKN smoothing, caches, Level 2/3,
decoding, quantization, training, and evaluation.

- **Common mistake:** its MariaDB migration and roadmap sections are not evidence that those
  backends/features exist in this checkout.

### [`studies/CONCEPT.md`](studies/CONCEPT.md)

Original architectural feasibility study and conceptual three-level DB-SLM blueprint.

- **Authority:** rationale only; current source and the v2 algorithm study own implementation facts.

### [`studies/ALGORITHMS_FLOW.md`](studies/ALGORITHMS_FLOW.md)

Compact training, inference, and Cheetah control-flow notes plus tracing knobs for performance work.

- **Common mistake:** future concurrency hooks listed there are planned ideas, not shipped paths.

### [`studies/BENCHMARKS.md`](studies/BENCHMARKS.md)

Authoritative repository record of dated, reproducible validation and performance runs.

- **Update rule:** include command, environment/scope, artifact paths, observed metrics, failures,
  and whether Cheetah or SQLite actually served the run.

### [`studies/best_commands.md`](studies/best_commands.md)

Repeatable smoke, queue-drain, and throughput command presets.

- **Common mistake:** validate presets against current CLI help before use and execute long variants
  inside a bounded session.

### [`studies/EvaluateSentenceQuality.md`](studies/EvaluateSentenceQuality.md)

Research notes for grammar, semantic acceptability, and quality scoring, including the path adopted
by the trainer.

- **Authority:** background research; [`src/db_slm/quality.py`](src/db_slm/quality.py) owns current
  scoring behavior.

### [`studies/notes/`](studies/notes/)

Historical author, TODO, platform, and command notes. `author.md` explains context-prototype intent;
`todo.md` and `win_cmd.md` preserve prior investigations and WSL recipes.

- **Status:** non-authoritative history. Promote active work to [`NEXT_STEPS.md`](NEXT_STEPS.md) and
  verified commands to [`README.md`](README.md) or [`studies/best_commands.md`](studies/best_commands.md).

### [`temp.txt`](temp.txt)

Tracked legacy snapshot of an older training script despite its `.txt` name.

- **Status:** not imported, executed, or authoritative. Do not implement changes here; use
  [`src/train.py`](src/train.py). Removal requires an explicit cleanup task because it is tracked
  user history.

## Features and Recurring Development Pitfalls

### Three-level database language model — Shipped, experimental quality

- **Behavior:** Level 1 supplies n-gram distributions, Level 2 adds conversation/cache/bias state,
  and Level 3 can select and verbalize a concept before decoding.
- **Flow and owners:** [`DBSLMEngine`](src/db_slm/pipeline.py) →
  [`level3.py`](src/db_slm/level3.py) → [`level2.py`](src/db_slm/level2.py) →
  [`decoder.py`](src/db_slm/decoder.py) / [`level1.py`](src/db_slm/level1.py).
- **Constraints:** core generation is database-native; optional embedding/quality models do not
  replace it.
- **Tests and gaps:** adapter and formatting paths have coverage; core statistical correctness and
  end-to-end quality do not.

### Streaming training, resume, and evaluation — Shipped

- **Behavior:** plain text and JSON/NDJSON inputs stream in chunks; JSON staging can use spawned
  workers; interrupted runs persist `var/train_resume.json`; periodic and chunk hold-out probes
  generate metrics and quality-queue entries. Level-3 staging previews are explicitly labeled as
  decoder input/reference pairs rather than generations. Periodic scheduling is bounded to one
  current probe per completed chunk even when the chunk crosses multiple token intervals.
- **Flow and owners:** [`src/train.py`](src/train.py) → dataset config/dependency layer →
  `DBSLMEngine.train_from_text` → `InferenceMonitor`/`EvalLogWriter`.
- **Constraints:** training startup requires `language_tool_python` and a working Java runtime;
  stdin is not resumable; resume rejects changed input fingerprints; metrics and resume files are
  runtime state.
- **Tests and gaps:** no focused resume/chunk/evaluation scheduler suite.

### Cheetah hot path, reducers, and prediction tables — Shipped

- **Behavior:** per-thread clients publish/fetch Level 1 data, page namespace scans, execute reducers
  through jobs, inspect system state, flush asynchronous mirror writes, and train/query/inherit
  prediction entries. MKN rebuilds compare current relational ingest counts with Cheetah and publish
  only changed follower sets before calculating probabilities.
- **Flow and owners:** [`src/db_slm/adapters/cheetah.py`](src/db_slm/adapters/cheetah.py) → Cheetah
  TCP command registry and reducers documented in [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md).
- **Constraints:** use a concrete client host, trusted network, named database isolation, bounded
  idle grace, and current payload serializers.
- **Tests and gaps:** Python adapter regressions plus extensive submodule tests; Python live-service
  integration is manual/smoke-only.

### Graph context memory — Shipped, opt-in and unmeasured

- **Behavior:** with `DBSLM_GRAPH_MEMORY=1` (or `--graph-memory`), training records each staged
  prompt/response pair as `ctx:<field>:<value>` and `term:<lemma>` nodes joined by
  `evokes`/`precedes`/`dep_<label>` edges, attaching the complete response sentence as a bounded node
  reference. Inference seeds `GRAPH_RECALL` with the turn's context values and content words, biases
  decoding toward the recalled terms, and widens the internal bias/embedding context with the
  hydrated sentences. `--graph-term-index-rebuild` pages `GRAPH_TERM_INDEX action=rebuild` after
  ingest so free-text seeds resolve.
- **Flow and owners:** [`src/train.py`](src/train.py) (`_configure_graph_memory`,
  `_observe_graph_memory`, `_rebuild_graph_term_index`) →
  [`graph_memory.py`](src/db_slm/graph_memory.py) → the `graph_*` surface of
  [`adapters/cheetah.py`](src/db_slm/adapters/cheetah.py); read path
  [`pipeline.py`](src/db_slm/pipeline.py) → [`decoder.py`](src/db_slm/decoder.py) →
  [`scoring.py`](src/db_slm/scoring.py).
- **Constraints:** requires the Cheetah backend; off by default. Per-record term/arc caps, a
  per-run node budget, and clamped recall `hops`/`branch_limit`/`budget`/`reference_limit` bound both
  the command stream and the payload. Graph output never enters the corpus or a visible response.
- **Tests and gaps:** protocol encoding, ingest shape, reference merging, and recall projection are
  covered in [`tests/test_graph_memory.py`](tests/test_graph_memory.py). No measurement of whether
  graph bias improves generation quality exists yet; that is an open item in
  [`NEXT_STEPS.md`](NEXT_STEPS.md).

### Adaptive tokenization and context signals — Shipped mechanism, experimental tuning

- **Behavior:** regex or Hugging Face tokenization, optional repeated-span merge tokens, grouped
  context penalties, sentence/window embeddings, tag-aware prototypes, and deepened prediction
  matrices influence training and sampling.
- **Flow and owners:** [`level1.py`](src/db_slm/level1.py) →
  [`context_dimensions.py`](src/db_slm/context_dimensions.py) →
  [`context_window_embeddings.py`](src/db_slm/context_window_embeddings.py) →
  [`decoder.py`](src/db_slm/decoder.py).
- **Constraints:** merge tokens activate only at n-gram order 5+; auto/persisted settings must agree
  across trainer and inference; offline embeddings use deterministic hashed guidance.
- **Tests and gaps:** tuning against GPTeacher and punctuation repetition remains in
  [`NEXT_STEPS.md`](NEXT_STEPS.md).

### Prompt framing and response guardrails — Shipped

- **Behavior:** dataset tags are registered atomically, every prompt terminates in a response tag,
  training/evaluation share before/after-prompt context lines, generated scaffold tokens remain
  banned through cache/prediction blending and retries, empty or `|END|`-only candidates are retried,
  internal concept/dependency context stays hidden, and output is consistently framed. Evaluation
  records the raw generation when one succeeds and a scaffold-free backstop only after every decoder
  attempt fails.
- **Flow and owners:** [`dataset_config.py`](src/db_slm/dataset_config.py) →
  [`prompt_tags.py`](src/db_slm/prompt_tags.py) → [`pipeline.py`](src/db_slm/pipeline.py) →
  [`text_markers.py`](src/db_slm/text_markers.py).
- **Tests and gaps:** formatter nesting is covered; full tag-ban exhaustion and lowercase-tokenizer
  behavior lack focused tests.

### Quality feedback and adversarial prediction updates — Shipped

- **Behavior:** evaluation combines overlap, ROUGE-L, perplexity, dependency alignment, structural
  diversity, cross-sample repetition, grammar, acceptability, and semantic metrics. Flagged samples
  enter a retraining queue; low-quality Cheetah contexts can reinforce reference tokens and
  down-weight generated tokens.
- **Flow and owners:** [`evaluation.py`](src/db_slm/evaluation.py) →
  [`quality.py`](src/db_slm/quality.py) / similarity helpers →
  `QualityGate` and [`AdversarialTrainer`](src/train.py).
- **Constraints:** retries and negative counts are capped; heavy scorers are load-gated; queue files
  may contain corpus text and remain local runtime data.
- **Tests and gaps:** metric math, queue schema, and adversarial updates lack focused tests.

### Smoke matrix and automatic queue drains — Shipped

- **Behavior:** scenario runs publish live benchmark state and metrics; queue depth can trigger a
  bounded drain that appends benchmark evidence and caps the queue after success.
- **Flow and owners:** [`Makefile`](Makefile) → [`scripts/smoke_train.py`](scripts/smoke_train.py) →
  [`scripts/drain_queue.py`](scripts/drain_queue.py).
- **Constraints:** workloads mutate databases/queues and MUST run in bounded sessions.
- **Tests and gaps:** script behavior has no automated test; use dry-run plus bounded smoke
  validation.

### Decoder scoring traces — Shipped library hook, CLI gap

- **Behavior:** callers can pass a `ScoreObserver` through `issue_prompt`/`DBSLMEngine.respond` to
  receive per-step base, penalty, cache, prediction, and final probabilities.
- **Flow and owners:** [`inference_shared.py`](src/db_slm/inference_shared.py) →
  [`pipeline.py`](src/db_slm/pipeline.py) → [`decoder.py`](src/db_slm/decoder.py) →
  [`scoring.py`](src/db_slm/scoring.py).
- **Known gap:** no supported `run.py` trace flag exists; the punctuation-collapse investigation in
  [`NEXT_STEPS.md`](NEXT_STEPS.md) must decide whether to add one.

### Pitfall: treating `--backonsqlite` as a working startup fallback

- **Symptom / wrong assumption:** documentation says an unreachable configured Cheetah backend can
  be bypassed by adding `--backonsqlite`.
- **Cause and invariant:** `build_cheetah_adapter` raises `SystemExit` before `train.main` can inspect
  the resulting adapter and honor the flag.
- **Risk area:** [`build_cheetah_adapter`](src/db_slm/adapters/cheetah.py) and engine construction in
  [`src/train.py`](src/train.py).
- **Safe pattern / regression check:** until fixed, choose `DBSLM_BACKEND=sqlite` explicitly for an
  authorized SQLite-only run; add a subprocess/startup regression before documenting automatic
  fallback.
- **Status:** active known bug.

### Pitfall: reintroducing prompt tags through retries or formatting

- **Symptom / wrong assumption:** the final retry candidate or an already-framed prompt leaks nested
  `|USER|:`, `|RESPONSE|:`, or aliases.
- **Cause and invariant:** retry exhaustion must clear rejected IDs, and wrapping must strip the
  terminal response tag before adding frames.
- **Risk area:** `DBSLMEngine.respond`, `_contains_prompt_artifacts`, and
  `TaggedResponseFormatter.wrap` in [`pipeline.py`](src/db_slm/pipeline.py).
- **Safe pattern / regression check:** preserve case normalization and run
  [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py).
- **Status:** fixed regression risk.

### Pitfall: reintroducing prompt tags through score mixtures

- **Symptom / wrong assumption:** a training probe prints an empty generation even though its
  n-gram candidates excluded `|USER|:`, `|RESPONSE|:`, or another structural tag.
- **Cause and invariant:** the prompt is intentionally present in the Level 2 cache; cache union or
  prediction blending can re-add a banned token unless the ban set is applied to every source. Three
  rejected candidates then collapse to the empty safe response.
- **Risk area:** `TokenScoringPipeline.score` in [`scoring.py`](src/db_slm/scoring.py).
- **Safe pattern / regression check:** filter cache and prediction distributions with `banned`
  before combining them; run [`tests/test_scoring.py`](tests/test_scoring.py).
- **Status:** fixed regression risk.

### Pitfall: accepting an end marker as a visible response

- **Symptom / wrong assumption:** a held-out evaluation sample is empty even though prompt tags stay
  banned and adjacent variants generate text.
- **Cause and invariant:** `|END|` is a valid decoder control token but becomes empty after
  `strip_end_marker`; candidate acceptance must validate the stripped visible text, retry an
  empty/marker-only decode, and keep the exhausted fallback independent of response scaffolding.
- **Risk area:** `DBSLMEngine.respond` in [`pipeline.py`](src/db_slm/pipeline.py).
- **Safe pattern / regression check:** strip the marker before accepting a candidate and run
  [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py).
- **Status:** fixed regression risk.

### Pitfall: training on serialized dependency records

- **Symptom / wrong assumption:** generated responses contain JSON-like `"lemma"`, `"dep"`,
  `"pos"`, `"head"`, or `"token"` fragments even though prompt tags remain banned.
- **Cause and invariant:** dependency arcs are structured evaluation/prediction metadata. Appending
  their JSON serialization after each response teaches the n-gram generator internal schema text;
  keep the objects on `EvaluationRecord` instead.
- **Risk area:** JSON/NDJSON staging in [`train.py`](src/train.py) and response acceptance in
  [`pipeline.py`](src/db_slm/pipeline.py).
- **Safe pattern / regression check:** keep dependency layers out of `CorpusChunk.train_text`,
  reject field-shaped legacy artifacts, and run [`tests/test_train_monitor.py`](tests/test_train_monitor.py)
  plus [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py).
- **Status:** fixed regression risk.

### Pitfall: building a `GRAPH_*` command by concatenating corpus text

- **Symptom / wrong assumption:** a node silently loses part of its id, or `GRAPH_NODE_SET` answers
  `ERROR,invalid_props`, even though the value looked fine in Python.
- **Cause and invariant:** Cheetah splits `GRAPH_*` arguments on whitespace, so a space truncates an
  id and breaks a JSON argument. Slugging and base64 encoding belong in the adapter.
- **Risk area:** the `_graph_*` helpers and `graph_*` methods in
  [`adapters/cheetah.py`](src/db_slm/adapters/cheetah.py), and `slugify`/`context_node_id`/
  `term_node_id` in [`graph_memory.py`](src/db_slm/graph_memory.py).
- **Safe pattern / regression check:** mint ids through `slugify`, let `_graph_token` reject a
  non-token value instead of truncating it, and run
  [`tests/test_graph_memory.py`](tests/test_graph_memory.py).
- **Status:** deliberate protocol boundary.

### Pitfall: erasing node references with a partial upsert

- **Symptom / wrong assumption:** a second training run leaves each node holding only the sentences
  from that run.
- **Cause and invariant:** `GRAPH_NODE_SET references=` **replaces** the stored list; omitting the
  argument preserves it, and `-` clears it. There is no server-side merge.
- **Risk area:** `GraphContextMemory._write_node` in [`graph_memory.py`](src/db_slm/graph_memory.py).
- **Safe pattern / regression check:** read the stored list back with `GRAPH_NODE_GET` on the first
  touch of a node in a process, merge under the 64-reference cap, and write the complete list; run
  `test_stored_references_are_merged_instead_of_overwritten`.
- **Status:** fixed regression risk.

### Pitfall: expecting a seed's own sentences back from recall

- **Symptom / wrong assumption:** `references=1` hydrates nothing even though the seeded context node
  demonstrably holds reference sentences.
- **Cause and invariant:** `GRAPH_RECALL` excludes seed nodes from the answer unless
  `include_seeds=1`, so the node the turn is *about* never returns its own provenance.
- **Risk area:** `GraphContextMemory.recall`/`signal_from_result` in
  [`graph_memory.py`](src/db_slm/graph_memory.py).
- **Safe pattern / regression check:** request `include_seeds=1` whenever references are hydrated and
  drop seed terms from the bias (they are the prompt, already in the Level 2 cache); run
  `test_recall_asks_for_seed_nodes_so_their_sentences_hydrate` and
  `test_seed_terms_contribute_sentences_but_never_bias`.
- **Status:** fixed regression risk.

### Pitfall: decoding reducer rows without removing storage transport

- **Symptom / wrong assumption:** count/probability/continuation payloads appear malformed even
  though the Cheetah reducer succeeded.
- **Cause and invariant:** reducer rows contain base64-wrapped stored bytes in addition to the
  DB-SLM serializer format.
- **Risk area:** `CheetahClient.decode_reduced_payload` and `CheetahHotPathAdapter.iter_*` in
  [`adapters/cheetah.py`](src/db_slm/adapters/cheetah.py).
- **Safe pattern / regression check:** decode transport first, serializer second; run the adapter
  tests.
- **Status:** fixed regression risk.

### Pitfall: modifying generic Cheetah behavior in the parent repository

- **Symptom / wrong assumption:** a server fix is copied into Python or a vendored Go file, leaving
  the upstream submodule and gitlink inconsistent.
- **Cause and invariant:** `cheetah-db/` is a separate Git repository and the owner of generic
  storage/protocol behavior.
- **Risk area:** [`.gitmodules`](.gitmodules), [`cheetah-db/`](cheetah-db/), and the Python adapter.
- **Safe pattern / regression check:** follow [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md), commit
  there, run its Go checks, then update the parent gitlink and Python compatibility tests.
- **Status:** deliberate repository boundary.

### Pitfall: relying on warm caches for correctness

- **Symptom / wrong assumption:** ingest → reducer → decode works before restart but loses mappings
  or reads stale payloads after restart/mutation.
- **Cause and invariant:** caches are derived; inserts/edits must seed them, deletes must invalidate
  them, and cold reads must reopen/reload canonical data.
- **Risk area:** `CheetahHotPathAdapter` cache helpers and Cheetah managed file/payload caches.
- **Safe pattern / regression check:** verify cold restart and round-trip behavior; prime performance
  separately with bounded scans, never as a correctness fix.
- **Status:** fixed regression risk with ongoing test responsibility.

### Pitfall: using legacy `run.sh` for unattended workloads

- **Symptom / wrong assumption:** sessions outlive the task indefinitely or cleanup depends on an
  attached parent shell.
- **Cause and invariant:** [`run.sh`](run.sh) has `sleep infinity` and no workload timeout.
- **Safe pattern / regression check:** use the bounded start/stop/smoke helpers and verify session,
  PID, timeout, and log ownership before launch.
- **Status:** deliberate legacy limitation.

## Interface Ownership Map

| Surface | Owner |
| --- | --- |
| `python3 src/train.py ...` | [`build_parser` / `main`](src/train.py) |
| `python3 src/run.py ...` and REPL commands | [`build_parser`, `PromptWorker`, `interactive_loop`](src/run.py) |
| Python library `DBSLMEngine` | [`src/db_slm/pipeline.py`](src/db_slm/pipeline.py) |
| Shared prompt API `issue_prompt` | [`src/db_slm/inference_shared.py`](src/db_slm/inference_shared.py) |
| SQLite schema and metadata | [`DatabaseEnvironment`](src/db_slm/db.py) |
| Dataset mapping/config surface | [`DatasetConfig`](src/db_slm/dataset_config.py) plus [`datasets/`](datasets/) |
| Token candidate scoring and trace API | [`src/db_slm/scoring.py`](src/db_slm/scoring.py) |
| Hot-path Python protocol | [`HotPathAdapter`](src/db_slm/adapters/base.py) |
| Cheetah TCP client, codecs, namespaces | [`src/db_slm/adapters/cheetah.py`](src/db_slm/adapters/cheetah.py) |
| Graph context memory conventions (ids, edges, seeds, recall projection) | [`src/db_slm/graph_memory.py`](src/db_slm/graph_memory.py) |
| Cheetah server command registry and on-disk formats | [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md) |
| Smoke matrix | [`Makefile`](Makefile) and [`scripts/smoke_train.py`](scripts/smoke_train.py) |
| Quality queue drain | [`scripts/drain_queue.py`](scripts/drain_queue.py) |
| Server/smoke lifecycle | [`scripts/start_cheetah_server.sh`](scripts/start_cheetah_server.sh), [`scripts/stop_cheetah_server.sh`](scripts/stop_cheetah_server.sh), [`scripts/start_cheetah_smoke_session.sh`](scripts/start_cheetah_smoke_session.sh) |

There are no HTTP routes, GUI screens, plugin hooks, packaged release APIs, or migration CLI in this
repository.

## Build, Run, Test, Debug, and Release

Prerequisites are Python 3.10+ (recommended in the README), pip, Java on `PATH` for LanguageTool, and
the initialized Cheetah submodule. Cheetah itself requires Go 1.24+ according to its nested
handbook. A spaCy English model is optional because dependency parsing can fall back to Stanza or
continue without annotations, but trainer startup does not proceed without LanguageTool/Java.

Bootstrap:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
bash cheetah-db/build.sh
```

The `cp` creates an ignored local configuration file; review host, port, and database before use.
Model-backed helpers may contact Hugging Face on first load. Set `DBSLM_EMBEDDER_OFFLINE=1` when
downloads are not allowed.

Start and stop the external service:

```bash
scripts/start_cheetah_server.sh
scripts/stop_cheetah_server.sh
```

Inspect the configured session and log while it runs. The start helper defaults to a 1,800-second
timeout and writes ignored PID/log state under `var/`.

Safe CLI inspection:

```bash
PYTHONPATH=src python3 src/train.py --help
PYTHONPATH=src python3 src/run.py --help
python3 scripts/smoke_train.py --dry-run
```

Representative bounded training/inference commands are maintained in
[`README.md`](README.md) and [`studies/best_commands.md`](studies/best_commands.md). Training writes
SQLite/Cheetah state, metrics, quality queue entries, and resume state; `--reset` deletes the chosen
SQLite and Cheetah database state.

Focused and full available Python checks:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cheetah_adapter -v
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_paraphraser_regression.py
```

Cheetah checks, run from the submodule after reading its handbook:

```bash
cd cheetah-db
go build ./...
go vet ./...
go test ./src
go test -race ./src
gofmt -l .
```

Use `gofmt -w` only on Go files intentionally changed. Benchmarks and live smoke tests are gated,
long-running, and MUST use the bounded-session discipline. The supported wrappers are:

```bash
scripts/start_cheetah_smoke_session.sh
make smoke-train
```

`make clean-smoke` is destructive: it removes named smoke SQLite files and `var/smoke_train`.

Debugging:

- `LMDB_LOG_LEVEL=3` enables verbose Python traces through
  [`src/log_helpers.py`](src/log_helpers.py).
- `CHEETAH_LOG_LEVEL=3` enables Cheetah command/reducer/trie traces.
- Trainer/run flags expose namespace summaries, `SYSTEM_STATS`, prediction probes, ingest
  profiling, and metrics export; use the parser/README for current spellings.
- Live benchmark status is written to `var/smoke_train/benchmarks.json`.

No Python package build, lint/format/typecheck configuration, CI workflow, security scanner, release
automation, database migration command, or deployment process is tracked. Do not invent a release
checklist; distribution is currently source plus the separately built Cheetah binary.

## Test Ownership Map

| Contract/subsystem | Focused check |
| --- | --- |
| Fixed-size Cheetah context/Top-K/probability/continuation payloads | `CheetahSerializerTests` in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) |
| Idempotent context publish and Top-K fetch | `CheetahHotPathAdapterTests` in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) |
| Reducer storage-base64 removal | `test_iter_counts_decodes_reducer_storage_transport` and `test_decode_reduced_payload_removes_insert_transport_layer` in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) |
| Cursored scan parsing and socket idle recovery | `CheetahClientParsingTests` in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) |
| Canonical `JOB` reducer flow and legacy alias fallback | `test_pair_reduce_uses_canonical_job_status_then_fetch` / `test_pair_reduce_falls_back_to_legacy_alias_without_job_api` in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) |
| Response frame de-nesting and scaffold stripping | `TaggedResponseFormatterTests` in [`tests/test_cheetah_adapter.py`](tests/test_cheetah_adapter.py) |
| Training/evaluation prompt context parity | [`tests/test_dataset_config.py`](tests/test_dataset_config.py) |
| Prompt-tag bans across cache/prediction/graph-bias mixing | [`tests/test_scoring.py`](tests/test_scoring.py) |
| Graph id slugging, protocol encoding, and clamped recall bounds | [`tests/test_graph_memory.py`](tests/test_graph_memory.py) |
| Graph ingest shape, reference merging, and recall projection | [`tests/test_graph_memory.py`](tests/test_graph_memory.py) |
| Evaluation writer avoids duplicate raw console samples | [`tests/test_evaluation.py`](tests/test_evaluation.py) |
| Evaluation events persist atomically while the run is active | [`tests/test_evaluation.py`](tests/test_evaluation.py) |
| Large-chunk periodic evaluation stays bounded | [`tests/test_train_monitor.py`](tests/test_train_monitor.py) |
| Marker-only raw decode retries and visible fallback | [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py) |
| Dependency records stay out of generation text and legacy artifacts are rejected | [`tests/test_train_monitor.py`](tests/test_train_monitor.py) + [`tests/test_pipeline_response.py`](tests/test_pipeline_response.py) |
| Current ingest counts replace stale Cheetah mirrors before smoothing | [`tests/test_level1_smoothing.py`](tests/test_level1_smoothing.py) |
| Paraphraser guard vs rewrite policy | [`scripts/run_paraphraser_regression.py`](scripts/run_paraphraser_regression.py) + [`studies/paraphraser_regression.jsonl`](studies/paraphraser_regression.jsonl) |
| Cheetah storage concurrency, trie, reducers, jobs, prediction tables, lifecycle, and formats | Test map in [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md) |

Known Python test gaps: SQLite schema upgrades/transactions; tokenizer and merging; MKN math and
quantization; context dimensions/windows; Level 2/3 persistence and restart; decoder sampling and numeric scoring
order; prompt-tag contamination retry exhaustion; dataset config parsing; trainer
reset/resume/chunking; evaluation metrics/retries;
quality queue/adversarial updates; multiprocessing REPL; shell lifecycle helpers; and live Python ↔
Cheetah integration. Add focused tests with changes in these areas rather than relying only on a
long smoke train.

## Data, Security, Privacy, and Compatibility Boundaries

- **Canonical vs derived:** source/config/studies are tracked. `.env`, raw corpora, SQLite/WAL/SHM,
  quality queues, metrics/logs, PID/resume files, Cheetah databases, model caches, and binaries are
  local runtime or derived data. Cheetah's named database is canonical for its own on-disk state;
  SQLite remains canonical for relational Level 2/3 records unless a feature explicitly mirrors
  metadata.
- **Secrets:** `.env` is ignored. Do not put credentials, private dataset content, or personal data
  into `.env.example`, logs committed to studies, test fixtures, or this handbook. No application
  credential store exists.
- **Network trust:** Cheetah's protocol is unauthenticated plaintext TCP and defaults to a broad
  listen address. Bind/expose it only on loopback or a trusted network; the Python client must use a
  connectable concrete address, never `0.0.0.0`.
- **Dataset privacy:** raw datasets and quality queue entries may contain user/corpus text and are
  ignored. Benchmark reports should aggregate or redact content unless a small fixture is explicitly
  safe to commit.
- **Destructive operations:** trainer `--reset`, `RESET_DB`, queue capping, `make clean-smoke`, and
  deleting SQLite/Cheetah directories can destroy state. Resolve exact paths and database names
  before execution; use isolated smoke namespaces.
- **Compatibility:** SQLite schema bootstrapping is additive/idempotent but has no formal migration
  or rollback framework. Token hashes, quantization, prompt tags, merge token strings, metadata
  keys, Cheetah namespaces, payload codecs, Absolute Vector Order, and graph node/edge id
  conventions are persisted compatibility surfaces. Changing `slugify` or the `ctx:`/`term:` id shape
  orphans every node written by an earlier run.
- **Backups:** no backup/restore workflow is automated. Preserve required SQLite files and Cheetah
  database directories before destructive format/schema work; do not treat the hot cache as a
  backup.
- **Limits:** dataset chunk sizes, maximum input lines, evaluation samples/variants/retries, response
  words, prediction negatives, reducer pages, worker pools, idle grace, and process timeouts exist
  to prevent resource runaway. New input paths and protocol payloads require equivalent validation.

## Current Status and Known Gaps

### Shipped

- End-to-end experimental Level 1/2/3 Python stack with training and spawned-worker inference CLIs.
- Regex and optional Hugging Face tokenization, auto n-gram order, merge tokens, grouped context
  penalties, tag-aware context prototypes, and Cheetah prediction-table integration.
- Dataset-configured prompt/response/context framing and prompt-tag leak guardrails.
- Parallel JSON/NDJSON staging, dependency metadata, resume state, periodic/hold-out evaluation,
  metrics export, quality queue, penalty tuning, and adversarial prediction updates.
- Cheetah thread-local adapter with namespace pagination, canonical job reducers, legacy aliases,
  fixed payloads, metadata mirrors, diagnostics, and verified concurrency repair in the pinned
  submodule.
- Opt-in graph context memory: corpus-to-graph observation during training, associative
  `GRAPH_RECALL` before decoding, bounded reference-sentence hydration, and a lexical seed index
  rebuild.
- Screen-first bounded Cheetah service/smoke helpers, smoke scenario telemetry, and automated queue
  drains.

### Experimental / Scaffold

- The repository as a whole remains experimental and explicitly lacks systematic human algorithm
  optimization.
- Generation quality, context-depth choices, merge significance, penalty tuning, deep prediction
  layers, and graph context memory are research mechanisms rather than stable production defaults.
  Graph memory has protocol and shape coverage but no measured effect on generation quality.
- Optional transformer-backed embeddings and CoLA scoring may download large models and degrade to
  unavailable/hashed behavior depending on host load and packages.
- Cheetah's simulated GPU prediction path, in-memory jobs, and other server-specific limitations are
  classified in [`cheetah-db/AGENTS.md`](cheetah-db/AGENTS.md).

<a id="known-gaps"></a>
### Known Gaps

- `--backonsqlite` does not recover from an unreachable configured Cheetah backend because adapter
  construction exits first. The README and CLI help must remain explicit about this until a startup
  regression proves a repaired flow.
- Python automated coverage is concentrated in the adapter/formatter; core statistical,
  persistence, evaluation, multiprocessing, and orchestration paths have the gaps listed in the
  test map.
- No tracked CI, linter, formatter, type checker, package/release pipeline, migration tool, or
  backup/restore workflow exists.
- [`run.sh`](run.sh) is a legacy unbounded launcher and is not compliant for unattended workloads.
- `temp.txt` is a tracked stale training snapshot that can confuse ownership.

### Near-Term Priorities

1. Run the evaluation-enabled 250/1,000-record Cheetah-only emotion scale-up and record decoder
   latency, Top-K hit ratio, and quality without SQLite fallback.
2. Measure graph context memory: compare quality metrics and decoder latency with
   `--graph-memory` on and off at the same scale, and record the ingest cost per chunk in
   [`studies/BENCHMARKS.md`](studies/BENCHMARKS.md) before recommending a default.
3. Validate deepened prediction layers against GPTeacher probes and measure punctuation repetition.
4. Use scoring traces to isolate punctuation collapse and decide whether the decoder needs a
   punctuation stage or a supported `run.py` trace flag.
5. Repair and test the explicit SQLite fallback contract, or remove the ineffective flag and stale
   documentation.

The authoritative, editable priority list is [`NEXT_STEPS.md`](NEXT_STEPS.md).

## Task Start and Handoff Checklist

Before editing:

1. Read this handbook and any nested handbook governing the target; check branch, status, submodule
   state, ignored/runtime boundaries, and recent relevant changes.
2. Use the linked source reference and interface map to identify the owner, callers, persistence or
   protocol boundary, and focused tests. Verify handbook claims against current code.
3. Read the relevant principle, critical contract, dataset config, algorithm study, or backlog item.
4. Decide which user docs, source-map entries, feature status, test map, data boundary, or roadmap
   facts the task can change.

Before handoff or commit:

1. Run the narrowest focused checks, then the full available suite proportional to risk. Run
   Cheetah build/vet/test/race checks inside the submodule for generic server changes.
2. Review the diff for accidental runtime data, secrets, stale paths, whitespace errors, and
   unrelated edits. Validate every local Markdown link touched.
3. Synchronize [`AGENTS.md`](AGENTS.md), [`README.md`](README.md),
   [`NEXT_STEPS.md`](NEXT_STEPS.md), dataset configs, studies, and benchmark evidence according to
   their ownership; remove superseded claims rather than appending a changelog.
4. Report commands run, commands not run, destructive/runtime effects, unresolved gaps, and any
   required submodule commit/gitlink relationship accurately.

### Handbook Update Triggers

| Change | Required update |
| --- | --- |
| Add, move, rename, split, or delete a meaningful file/symbol | Update its linked source subsection, callers, tests, interface map, and pitfalls. |
| Change a CLI, library API, dataset mapping, protocol, reducer, or prediction operation | Update critical contracts, interface ownership, README, and focused tests. |
| Change settings, dependency, prerequisite, platform behavior, or defaults | Update `.env.example`/requirements, build/run guidance, README, and feature constraints. |
| Change schema, metadata, namespace, payload, cache, reset, or compatibility behavior | Update persistence/data boundaries, migration/rollback note, restart/round-trip tests, and destructive warnings. |
| Change concurrency, lifecycle, performance, or observability | Update principles/contracts, architecture flow, helper commands, and benchmark guidance. |
| Fix/discover a reusable failure mode | Add or consolidate a pitfall; keep active defects under Known Gaps and active work in `NEXT_STEPS.md`. |
| Complete roadmap or research work | Move it to Shipped only after code and verification; record reproducible evidence in `studies/`. |
| Pure refactor or typo | Update only paths, owners, commands, or meaning that actually changed; do not manufacture handbook churn. |
