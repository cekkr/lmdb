PYTHON ?= python3
SMOKE_DB ?= var/smoke-train.sqlite3

.PHONY: smoke-train clean-smoke

smoke-train:
	$(PYTHON) src/train.py datasets/emotion_data.json \
		--db $(SMOKE_DB) \
		--reset \
		--json-chunk-size 120 \
		--max-json-lines 400 \
		--eval-interval 1500 \
		--eval-samples 2 \
		--eval-pool-size 40 \
		--profile-ingest
	$(PYTHON) src/run.py \
		--db $(SMOKE_DB) \
		--prompt "Summarize how the DB-SLM handles short validation runs." \
		--user smoke-test \
		--agent db-slm

clean-smoke:
	rm -f $(SMOKE_DB)
