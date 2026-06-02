# src/data_handler.py

from pathlib import Path
import sys
import numpy as np
import pandas as pd

import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.seeder import run as run_seeder
from src.config import (
    SEEDER_FILE,
    DATASET_FILE,
    SAMPLE_FILE,
    SEED,
    DATA_TEMP_SPLIT, HALF_SPLIT
)

def _filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove "N/A" from Education Level, "NaN" from Marital Status and "abc" from the Income Category
    :return:
    """
    # Remove "N/A" from Education Level, "NaN" from Marital Status and "abc" from the Income Category
    for col in ['education_level', 'marital_status', 'income_category']:
        df[col] = df[col].replace('N/A', np.nan)
        df[col] = df[col].replace('NaN', np.nan)
        df[col] = df[col].replace('abc', np.nan)

    return df

def _drop(x_train: pd.Series, y_train: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Drop rows in X_train and y_train where y_train has NaN values
    y_train = y_train.dropna()
    x_train = x_train.loc[y_train.index] # Keep only rows in X_train that match y_train's index

    if y_train.empty:
        print('❗ Warning! Target training data is empty after dropping NaNs. Imputation cannot be performed. ❗')
    else:
        y_train = y_train.fillna(y_train.mode()[0]) # Impute only if y_train is not empty

    return x_train, y_train

def _merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper. Changed from __merge to _merge for better
    idiomatic consistency and ease of testing.
    """
    df_seeder = pd.read_csv(Path(SEEDER_FILE))
    df_combined = pd.concat([df, df_seeder]).drop_duplicates().reset_index(drop=True)
    df_combined.to_csv(DATASET_FILE, index=False)

    return df_combined


def _split_data(df: pd.DataFrame) -> dict:

    # Dependent (Target) variable
    target = df['attrition_flag']

    # Independent variables
    df_independent = df.drop('attrition_flag', axis=1)

    # --- Split data into 70% training data and 30% temporary data --- #
    x_train, x_temp, y_train, y_temp = train_test_split(
        df_independent,
        target,
        test_size=DATA_TEMP_SPLIT,
        random_state=SEED,
        stratify=target
    )

    # --- Then take the remaining temporary data 30% and split in half --- #
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=HALF_SPLIT,
        random_state=SEED,
        stratify=y_temp
    )

    print('\n--- (Split) Data Shapes ---')
    print(f'Shape of X training: {x_train.shape}')
    print(f'Shape of Y training: {y_train.shape}')
    print(f'Shape of X validation: {x_val.shape}')
    print(f'Shape of Y validation: {y_val.shape}')
    print(f'Shape of X testing: {x_test.shape}')
    print(f'Shape of Y testing: {y_test.shape}')

    print('\n--- (Split) Data Types ---')
    print(f'Data types of X training:\n{x_train.dtypes}')
    print(f'Data type of Y training: {y_train.dtype}')

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test
    }


class DataHandler:
    def __init__(self, use_seeder: bool=False):
        self.data_path = SAMPLE_FILE
        self.use_seeder = use_seeder
        self.data = {}
        self.get()

    def get(self):
        """
        Public API to orchestrate the data pipeline.
        """
        df = self.load_data()
        df = _filter(df)
        raw_data = _split_data(df)
        self.data = raw_data.copy()
        self._impute_data()
        self._encode_data()
        
        self.data['x_train'], self.data['y_train'] = _drop(self.data['x_train'], self.data['y_train'])

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        if self.use_seeder:
            # creates seeder file: data/seeder_data
            run_seeder()
            # Merge seeder data with df here, if needed.
            df = _merge(df)

        if df is None:
            print(f'Error! {self.data_path} not found or empty!')
            sys.exit(1)
        else:
            return df

    def _impute_data(self) -> None:
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

        csv_columns = ['education_level', 'marital_status', 'income_category']
        data_columns = ['x_train', 'x_val', 'x_test']

        for col in data_columns:
            self.data[col][csv_columns] = imputer.fit_transform(self.data[col][csv_columns])

    def _encode_data(self):
        # # Find categorical features
        numerical_labels = {'Existing Customer': 0, 'Attrited Customer': 1}
        categorical_cols = self.data['x_train'].select_dtypes(include=['object']).columns

        _encoder = ce.OrdinalEncoder(cols=categorical_cols)

        self.data['x_train'] = _encoder.fit_transform(self.data['x_train'])
        for col in ['x_val', 'x_test']:
            self.data[col] = _encoder.transform(self.data[col])

        # Convert target variable to numerical labels
        for col in ['y_train', 'y_val', 'y_test']:
            self.data[col] = self.data[col].map(numerical_labels)

        # Assume you have a y value and convert it as well
        #self.data['y_val'] = self.data['y_val'].map(numerical_labels)
        #self.data['y_test'] = self.data['y_test'].map(numerical_labels)
