# src/preprocessing.py
import pandas as pd

from src import seeder
from src import config

def load_data(seed_data: bool = False) -> pd.DataFrame:
    """Loads data from a CSV file."""

    path = config.SOURCE_FILE
    df = pd.read_csv(path)

    # Should we merge with seeder?
    if seed_data:
        # Run the seeding.py to create the seeder data, then merge it.
        seeder.run()
        return merge_seeder_data(df)


    return df

def merge_seeder_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

def split_data():
    pass