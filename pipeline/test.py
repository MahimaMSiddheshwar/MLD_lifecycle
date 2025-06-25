import warnings
import pandas as pd
from zenml import pipeline
from src.Stage_1_Ingestion.data_loaders import dataLoader, dataCheck
from src.Stage_2_EPD_Analysis.PED_Analysis import UnifiedPEDAnalyze
from src.Stage_3_Split_data.data_split import data_splitter, baseline
from src.Stage_4_Preprocessor.preprocessor import data_preprocessor

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
    train, test, val = data_preprocessor(train, test, val)
