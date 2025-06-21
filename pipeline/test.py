from src.Stage_1_Ingestion.data_loaders import dataLoader, dataCheck, PEDAnalyze
import warnings
from zenml import pipeline

warnings.filterwarnings("ignore")


@pipeline
def main():
    # Load the data
    data_loader = dataLoader(
        "Data/merged_all_3_datasets.csv", "Breast Cancer")
    # TODO: Add parellel processing for data Analysis
    dataCheck(data_loader)
    print("Data health check completed successfully.")
    PEDAnalyze(data_loader)
