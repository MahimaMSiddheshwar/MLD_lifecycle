.PHONY: all prep eda features split baseline

all: prep eda features split baseline

prep:
		python -m data_cleaning.data_preparation --target is_churn

eda:
		python -m data_analysis.EDA --mode all --target is_churn --profile

features:
		python -m feature_engineering.feature_engineering --data data/processed/scaled.parquet --target is_churn

split:
		python -m data_cleaning.split_and_baseline --target is_churn --seed 42 --stratify

baseline:
		# Already part of split stage
		@echo "Baseline metrics are in reports/baseline/baseline_metrics.json"
