import warnings
import pandas as pd
from zenml import pipeline
from src.Stage_1_Ingestion.data_loaders import dataLoader, dataCheck
from src.Stage_2_EPD_Analysis.PED_Analysis import UnifiedPEDAnalyze
from src.Stage_3_Split_data.data_split import data_splitter, baseline
from src.Stage_4_Preprocessor.preprocessor import missing_imputer, outlier_detector
from src.utils.PipelineReporter import PipelineReporter

warnings.filterwarnings("ignore")


@pipeline
def main():
    # Load the data
    data_loader = dataLoader(
        "Data/merged_all_3_datasets.csv", "Breast Cancer")
    # TODO: Add parellel processing for data Analysis
    dataCheck(data_loader)
    print("Data health check completed successfully.")
    UnifiedPEDAnalyze(data_loader)
    train, test, val = data_splitter(data=data_loader)
    baseline(train, pd.concat(test, val))
    print("Data split and baseline model training completed successfully.")

    reporter = PipelineReporter(max_charts=50)

    # Register components with `.report` or `.get_pipeline_report()` method
    train, test, val = missing_imputer(train, test, val)
    reporter.register("missing_imputer", missing_imputer)
    train, test, val = outlier_detector(train, test, val)
    reporter.register("outlier_detector", outlier_detector)

    # Generate Markdown + HTML + JSON reports and log to MLflow
    reporter.generate_report(output_name="final_pipeline_report")
