# Credit Card Churn Predictor

![Hero Image](data/hero.webp)

A comprehensive machine learning pipeline designed to predict whether a credit card customer will churn (attrite) or be retained. This project employs various ensemble learning techniques and strategies for handling imbalanced datasets to maximize prediction accuracy and recall.

## Features

*   **Data Pipeline**: Automated loading, cleaning, imputation, and encoding of data.
*   **Exploratory Data Analysis (EDA)**: Built-in mode to visualize feature distributions and correlations.
*   **Imbalanced Data Handling**:
    *   Synthetic Minority Over-sampling Technique (SMOTE)
    *   Random Under Sampling
*   **Model Zoo**:
    *   Bagging Classifier
    *   Random Forest
    *   AdaBoost
    *   Gradient Boosting
    *   XGBoost
*   **Optimization**: Hyperparameter tuning using `RandomizedSearchCV`.
*   **Synthetic Data Generation**: Utility to generate seed data for testing purposes.

## Project Structure

```text
credit-card-churn-predictor/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── data/                   # Data storage (sample_data.csv, etc.)
│   ├── dataset.csv        # Sample dataset for training
│   ├── hero.webp          # Hero image for project
├── models/
│   ├── ada_boost.py         # Model definition for ADA Boost Classifier
│   ├── bagging.py           # Model definition for Bagging Classifier
│   ├── gradient_boosting.py # Model definition for Gradient Boosting Classifier
│   ├── model_evaluator.py   # Model definition and training
│   ├── random_forest.py     # Model definition for Random Forest Classifier
│   ├── xg_boost.py          # Model definition for XGBoost Classifier
├── outputs/
├── src/
│   ├── config.py           # Configuration constants
│   ├── eda.py              # Exploratory Data Analysis logic
│   ├── data_handler.py     # Data cleaning and transformation
│   ├── seeder.py           # Synthetic data generator
│   └── utils.py            # Helper functions and plotting
└── venv/                    # Virtual environment
```

## Installation

1.  **Clone the repository** and navigate to the project directory.

2.  **Set up a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is controlled via `main.py` with command-line arguments.

### 1. Train Models (Default)
Runs the full machine learning pipeline: data splitting, preprocessing, model training (Original, Oversampled, Undersampled), hyperparameter tuning, and final evaluation.

```bash
python main.py
```


If you want to seed a sample data.
```bash
./venv/bin/python main.py --seed
```

Run Exploratory Data Analysis (EDA) Pipeline

```bash
./venv/bin/python main.py --mode eda
```
