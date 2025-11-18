## Doubts

### How works context embedding during training and evaluation:
Context dimensions here are trained “on the fly” via ContextWindowEmbeddingManager (see src/db_slm/context_window_embeddings.py). Each time DBSLMEngine.train_from_text() ingests a corpus chunk (src/db_slm/pipeline.py (line 230)), it calls context_windows.observe_corpus(corpus) before tokenization. That call:

Uses ContextWindowExtractor to slice the raw text into fixed-size windows for every configured dimension (sample count is controlled by max_train_windows, default 24; stride ~50% of the window length).
Sends those windows through the SentencePartEmbeddingPipeline’s embedder (default all-MiniLM-L6-v2, switchable via DBSLM_EMBEDDER_MODEL).
Updates a running prototype per dimension (ContextDimensionPrototype), keeping the mean embedding vector plus optional tag statistics (when context tags map to enumerated indices).
Stores the prototypes in SQLite metadata (and via cheetah hot-path when available) so later runs can reuse them.
No evaluation cycle is required for this learning step—the embeddings are collected during training-script ingestion itself, alongside n‑gram counts. During inference, ContextWindowEmbeddingManager.weights_for_text() samples new windows from the live conversation history, embeds them with the same model, compares them to the stored prototypes via cosine similarity, and outputs per-dimension weight multipliers (scaled 0.5–1.5, tag-aware). Those weights feed into the decoder’s presence/frequency penalties so each context dimension fine-tunes token generation.

So we’re not comparing LMDB predictions against the pre-trained model at eval time; instead, the pre-trained embedder just supplies a continuous representation for each dimension during both ingest and inference. The actual LM adjusts token-by-token penalties using those learned prototypes.