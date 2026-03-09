import pandas as pd
from pathlib import Path


def test_processed_data_exists():

    data_path = Path("data/processed/AAPL.parquet")

    assert data_path.exists(), "Processed data file missing"


def test_dataframe_not_empty():

    df = pd.read_parquet("data/processed/AAPL.parquet")

    assert len(df) > 0, "Dataframe is empty"
