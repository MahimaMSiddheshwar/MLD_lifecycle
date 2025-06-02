# Makefile

.PHONY: all ingest prepare eda eda-adv eda-probabilistic feature split train clean

all: ingest prepare eda eda-adv eda-probabilistic feature split train

# ───────────────────── Phase 2 ─────────────────────
ingest:
	dvc repro ingest

# ───────────────────── Phase 3 ─────────────────────
prepare:
	dvc repro prepare

# ─────────── Phase 4 core & advanced EDA ───────────
eda:
	dvc repro eda_core

eda-adv:
	dvc repro eda_advance

eda-prob:
	dvc repro eda_probabilistic

# ───────────────────── Phase 5 ─────────────────────
feature:
	dvc repro feature_eng

# ──────────────────── Phase 5½ ────────────────────
split:
	dvc repro split_and_baseline

# ───────────────────── Phase 6 ─────────────────────
train:
	dvc repro train_model

clean:
	dvc gc --workspace --force
