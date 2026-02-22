# src/preprocessing.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from src import seeder
from src.config import *


def load_data(seed_data: bool = False) -> pd.DataFrame:
    """Loads data from a CSV file."""

    path = SAMPLE_FILE
    df = pd.read_csv(path)

    # Should we merge with seeder?
    if seed_data:
        # Run the seeding.py to create the seeder data, then merge it.
        seeder.run()
        return merge_seeder_data(df)

    return df

def merge_seeder_data(df_sample: pd.DataFrame) -> pd.DataFrame:
    df_seeder = pd.read_csv(SEEDER_FILE)
    df_combined = pd.concat([df_sample, df_seeder]).drop_duplicates().reset_index(drop=True)
    df_combined.to_csv(DATASET_FILE, index=False)

    return df_combined


def split_seeder_data(df: pd.DataFrame):
    # INDEPENDENT VARIABLES
    df_independent = df.drop('attrition_flag', axis=1)

    # DEPENDENT VARIABLE
    df_target = df['attrition_flag']

    # --- Split data into 70% training data and 30% temporary data --- #
    x_training_data, x_temp_data, y_training_data, y_temp_data = train_test_split(
        df_independent,
        df_target,
        test_size=DATA_TEMP_SPLIT,
        random_state=SEED,
        stratify=df_target
    )

    # --- Then take the remaining temporary data 30% and split in half --- #
    x_validation_data, x_testing_data, y_validation_data, y_testing_data = train_test_split(
        x_temp_data,
        y_temp_data,
        test_size=HALF_SPLIT,
        random_state=SEED,
        stratify=y_temp_data
    )

    return x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Remove "N/A" from Education Level, "NaN" from Marital Status and "abc" from the Income Category
    df['education_level'] = df['education_level'].replace('N/A', np.nan)
    df['marital_status'] = df['marital_status'].replace('NaN', np.nan)
    df['income_category'] = df['income_category'].replace('abc', np.nan)

    return df
