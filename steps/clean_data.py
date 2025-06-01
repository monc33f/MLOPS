from zenml import step
from typing import Tuple
import logging
import pandas as pd

@step
def clean_data(
    tr_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ts_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logging.info("Nettoyage des datasets d'entraînement, validation et test.")


    return tr_df, valid_df, ts_df
