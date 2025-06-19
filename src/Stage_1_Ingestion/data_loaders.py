import pandas as pd
from zenml import step
from DataCollector import DataCollector
from DataHealthCheck import DataHealthCheck

DATASET_TARGET_COLUMN_NAME = "target"


@step
def data_loader(file: str = None, project: str = "Default") -> pd.DataFrame:
    dataCollector = DataCollector(suite_name=project)
    data = dataCollector.read_file(file)
    dataHealthCheck = DataHealthCheck()
    dataHealthCheck.run_all_checks()
    data_integrity_report = dataHealthCheck.data_integrity_checker()
    return data, data_integrity_report
