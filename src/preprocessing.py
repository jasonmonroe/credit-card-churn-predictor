# src/preprocessing.py

import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.impute import SimpleImputer
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


def impute_missing_values(x_training_data, x_validation_data, x_testing_data):
    impute_columns = ['education_level', 'marital_status', 'income_category']
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

    # Fit and transform the train data
    x_training_data[impute_columns] = imputer.fit_transform(x_training_data[impute_columns])

    # Transform the validation data
    x_validation_data[impute_columns] = imputer.transform(x_validation_data[impute_columns])

    # Transform the test data
    x_testing_data[impute_columns] = imputer.transform(x_testing_data[impute_columns])

    return x_training_data, x_validation_data, x_testing_data


def encode_data(x_training_data, x_validation_data, x_testing_data, y_training_data, y_validation_data, y_testing_data):
    # Find categorical features
    numerical_labels = {'Existing Customer': 0, 'Attrited Customer': 1}
    categorical_cols = x_training_data.select_dtypes(include=['object']).columns

    encoder = ce.OrdinalEncoder(cols=categorical_cols)

    # Fit and transform the training data
    x_training_data = encoder.fit_transform(x_training_data)

    # Transform the validation data
    x_validation_data = encoder.transform(x_validation_data)

    # Transform the test data
    x_testing_data = encoder.transform(x_testing_data)

    # Convert target variable to numerical labels
    y_training_data = y_training_data.map(numerical_labels)

    # Assuming you have a y_val, convert it as well
    y_validation_data = y_validation_data.map(numerical_labels)
    y_testing_data = y_testing_data.map(numerical_labels)

    return x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data
