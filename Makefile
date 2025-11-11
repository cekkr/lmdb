PYTHON ?= python3
SMOKE_SCENARIOS ?=
SMOKE_DATASET ?=
SMOKE_MATRIX ?=
SMOKE_BENCH ?=

.PHONY: smoke-train clean-smoke

SMOKE_ARGS :=
SMOKE_ARGS += $(if $(SMOKE_SCENARIOS),--scenarios $(SMOKE_SCENARIOS),)
SMOKE_ARGS += $(if $(SMOKE_DATASET),--dataset $(SMOKE_DATASET),)
SMOKE_ARGS += $(if $(SMOKE_MATRIX),--matrix $(SMOKE_MATRIX),)
SMOKE_ARGS += $(if $(SMOKE_BENCH),--benchmarks $(SMOKE_BENCH),)

smoke-train:
	$(PYTHON) scripts/smoke_train.py $(SMOKE_ARGS)

clean-smoke:
	rm -f var/smoke-train-baseline.sqlite3 var/smoke-train-penalty.sqlite3
	rm -rf var/smoke_train
