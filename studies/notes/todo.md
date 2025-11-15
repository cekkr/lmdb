# Step 1

- The tokenization is too arbitrary especially in punctuation splitting, in fact results use to have difficulties in puntactions writing while generating.
- In @sentence_parts.py, there is another arbitrary forcing of words: _EMOTION_WORDS. Why having a dataset if you force the words? Remove any ARBITRARY reference.
- Try to add the support to an external and well established tokenizator, to use overall during training and in-training evaluations.

Let's take this "default train command":
python3.11 src/train.py datasets/emotion_data.json --ngram-order 4 --recursive --reset --json-chunk-size 50 --eval-samples 2 --eval-pool-size 5000 --chunk-eval-percent 10.0 --eval-interval 5000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 1-4,5-13
Is it resetting the cheetah db? Is it using an ad hoc database name for this training not using the default one?
The penalty parameters ar respected seen the too many repeatitions?

Then, try to make context-dimension parameter better, with more dimensions instead of simple "word-sentence". It should be more "irrational" and with support of more dimension.
Example --context-dimension 4,8,4 (3 dimension, the first of 4 vectors, the second 8 and the third 4)

# Step 2
- (x) Move any AI_REFERENCE.md directive and documentation exclusively about cheetah-db inside cheetah-db/ directory, but remember to @AI_REFERENCE.md to check cheetah-db/AI_REFERENCE.md when working with cheetah.
- (x) Update @README.md and @BEST_COMMANDS.md to document and show examples about CURRENTLY EFFECTIVELY VALID commands arguments, especially about @train.js (There was another thing but I forgot it)
- (x) Add support, in dataset samples preparation during first steps of a cycle, take advantages of Dependency Parsing to create an additional layer (and fast categorizations) of words to use as "strong tokens group reference" with additional context to the series of dependencies enumeration as array. This would help heavily training and evaluation process, helping to reduce also ngram-order number. You can use spaCy or Stanza
- (x) Remove the direct reference to "Emotional Embeddings" et simili in src/ code: this is a reference of the current dataset, so you have to transform it in a generic "additional context token". For linearity, better create in datasets/ the config file dataset_name.config.json, in this case emotion_data.config.json . Let's take as example a sample of emotion_data.json dataset: {"prompt": "Examine how Envy plays a role in leadership, management, and organizational behavior.", "emotion": "Envy", "response": "Envy is a complex emotion [...]"}. The json should make evident that "prompt" is the user request and "response", obviously, the response part. How response should be handled that currently is directly integrated in src/ code (but now should be made generical), and that "emotion" is an additional token context to enumerate ("tokenization"). And similar ad hoc configuration that could change from dataset to dataset.
- (x) Improve cheetah-db/ multithreading management: first of all, detect the available cores in current machine and allow real time CPU and I/O usage in general and in precision by the server itself. This is useful for the next step: improve the multithreading management of multiple connections but also of a single connection, in basis of current resources availability and realtime usage (to avoid CPU/RAM/HD bottlenecks), take advantages of multiple file division of server for improving multithreading on different cores, especially considering the caching advantages.

# Step 3
- (x) Pre-check requirements before execution of @train.py: for example, after hours of execution, I discovered that I can't use python-language-tools due to the lack of Java. And these kind of error should be "exit" (blocking): are needed tools for a correct training, so if not installed train.py shouldn't run and explain why.
- (x) During the composition of metrics json log during training, evaluate a complete sentence (not truncated to early) to add as reference of the cycle.

# WebGPU in Python
(No advantages because:)

src/train.py itself delegates all heavy lifting to DBSLMEngine plus spaCy/Stanza preprocessing and the evaluation helpers. These workloads are either database/IO-bound (SQLite + cheetah RPCs inside db_slm modules) or rely on third-party CPU stacks. None expose hooks where a custom WebGPU kernel could be slotted in, so adding wgpu would mean rewriting big chunks of the existing engine rather than flipping an accelerator flag.

The only tensor-heavy pieces imported by train.py live under src/db_slm/sentence_parts.py (lines 150-199) (SentenceTransformer embedder) and src/db_slm/quality.py (lines 80-214) (Hugging Face CoLA head). Both already ride on PyTorch and automatically use CUDA/MPS when present, so they already have GPU acceleration paths. WebGPU cannot be enabled there without reimplementing those models with a completely different backend, because PyTorch/transformers/sentence-transformers have no WebGPU execution provider.

src/run.py plus its imports (db_slm.decoder, db_slm.level*, etc.) focus on orchestrating SQLite queries and the cheetah-db RPC adapter. The hot paths are SQL statements and Python control flow; there are no custom numeric kernels or vectorizable tensor ops to port to WebGPU.

Other imported helpers (helpers/char_tree_similarity.py, helpers/resource_monitor.py, db_slm/metrics.py, etc.) are light-weight scalar computations and logging. Moving those to WebGPU would add significant complexity for negligible speedups.

Given the current architecture, there isn’t a part of train.py, run.py, or their dependency graph that can practically “take advantage” of a WebGPU rewrite without overhauling core components or replacing mature third-party libraries that already provide CUDA/MPS acceleration.