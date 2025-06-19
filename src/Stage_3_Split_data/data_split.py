from ThreeWaySplit import SplitThreeWay
from typing_extensions import Annotated
from zenml import step
from typing import Tuple
import pandas as pd
from BaselineModel import AutoBaseline

DATASET_TARGET_COLUMN_NAME = ""


@step
def data_splitter(
    data: pd.DataFrame,
    target: str = DATASET_TARGET_COLUMN_NAME,
    stratify: bool = True,
    oversample: bool = True,
    seed: int = 42,
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "test"]]:
    train, test, val = SplitThreeWay(
        data=data,
        stratify=stratify,
        seed=seed,
        oversample=oversample,
        target=target)

    baselineModel = AutoBaseline(target=DATASET_TARGET_COLUMN_NAME)
    baselineModel.run(train, pd.concatenate(test, val))

    return train, test, val
