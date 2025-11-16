## emotion_data.json (Kaggle — Emotional Intelligence Booster Zephyr 7B)

- Rows ingested: ~287k (JSONL).
- Prompt length: 11.9 words on average (std-dev dominated by short directive stems).
- Response length: 347 words on average, median 336, 90th percentile 437, max 1,251 words.
- Only 0.01% of responses fit inside 128 words, which explains previous `|RESPONSE|` truncation in evaluation logs.
- Emotion labels skew toward the primary Plutchik wheel but still include calmer tags (calm, gratitude, curiosity).

### Notes for Training / Evaluation

- Because responses are long-form, `train.py` now derives `min_response_words` from the reference length (capped at 512 words) so evaluation probes can cover most of the answer before logging metrics.
- The quality gate under `var/eval_logs/quality_retrain_queue.jsonl` captures low-quality generations (grammar errors ≥3, CoLA acceptability <0.45, semantic similarity <0.55, or length mismatch >40%) for targeted retraining slices.
- When profiling the dataset, prefer chunk sizes ≥2,000 rows so the sentence-part embedder can observe enough punctuation statistics to avoid aggressive truncation.
- Field mapping + additional context tokens are declared in `datasets/emotion_data.config.json`. `train.py` reads this to emit the `Emotion` header along with the canonical `|CTX|:emotion:<value>` tags so other datasets can define their own schema without touching source code.

Source: [https://www.kaggle.com/datasets/anthonytherrien/emotional-intelligence-booster-zephyr-7b](https://www.kaggle.com/datasets/anthonytherrien/emotional-intelligence-booster-zephyr-7b)
Fast download [https://drive.google.com/file/d/140Jbciw_ulNemJhtDSI4igZgTGQaXPjf/view?usp=sharing](https://drive.google.com/file/d/140Jbciw_ulNemJhtDSI4igZgTGQaXPjf/view?usp=sharing)

## GPTeacher
Dataset URL: [https://github.com/cekkr/GPTeacher](https://github.com/cekkr/GPTeacher) merge all datasets using `node merge.js`
Direct download: [https://raw.githubusercontent.com/cekkr/GPTeacher/refs/heads/main/GPTeacher.json](https://raw.githubusercontent.com/cekkr/GPTeacher/refs/heads/main/GPTeacher.json)