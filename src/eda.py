# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    histogram_boxplot,
    labeled_barplot,
    stacked_barplot,
    distribution_plot_wrt_target,
    show_banner
)
from src.config import *

def run_eda(df: pd.DataFrame):
    # How is the total transaction amount distributed?
    trans_amt_dist = df['total_trans_amt'].describe().T

    print('Distribution of total transaction amount:')
    for key, value in trans_amt_dist.items():
        print(f"{key} is {value}")

    # What is the distribution of the level of education of customers?
    edu_level_dist = df['education_level'].value_counts()
    sum_val = edu_level_dist.sum()

    show_banner('Distribution of education level')
    for key, value in edu_level_dist.items():
        pct = round((value / sum_val) * PERCENTILE)
        print(f"{key} is {value} or {pct}%")

    # Display box plot
    labeled_barplot(df, 'education_level', 'Customers by Education Level')

    # What is the distribution of the level of income of customers?

    income_dist = df['income_category'].value_counts()
    sum_val = income_dist.sum()

    show_banner('Distribution of income level')

    for key, value in income_dist.items():
      pct = round((value / sum_val) * PERCENTILE)
      print(f"{key} is {value} or {pct}%")

    distribution_plot_wrt_target(df, 'income_category', 'attrition_flag')

    # Display labeled bar plot
    labeled_barplot(df, 'income_category', 'Customers by Income Range')

    # How does the change in transaction amount between Q4 and Q1 (total_ct_change_Q4_Q1) vary by the customer's account status (Attrition_Flag)?
    change_data = df.groupby(['attrition_flag'])['total_ct_chng_q4_q1'].describe()

    print(change_data)

    # Display box plot
    histogram_boxplot(df, 'total_ct_chng_q4_q1', '(Q4) Total Customer Count Change Histogram')

    # How does the number of months a customer was inactive in the last 12 months (Months_Inactive_12_mon) vary by the customer's account status (Attrition_Flag)?
    inactive_data = df.groupby('attrition_flag')['months_inactive_12_mon'].describe()

    show_banner('Inactive Data')
    print(inactive_data)

    histogram_boxplot(df, 'months_inactive_12_mon', 'Last 12 months of customer inactivity')
    histogram_boxplot(df, 'customer_age', 'Ages of Customers Histogram')
    histogram_boxplot(df, 'months_on_book', 'Months on the books Histogram')
    histogram_boxplot(df, 'total_relationship_count')
    histogram_boxplot(df, 'months_inactive_12_mon', 'Inactive customers of at least 1 year')
    histogram_boxplot(df, 'contacts_count_12_mon', 'Customer contacts for the last 12 months')

    # Count the number of contacts by highest frequency
    print(df['contacts_count_12_mon'].value_counts(1).sort_values(ascending=False))

    histogram_boxplot(df, 'credit_limit', 'Customer Credit Limits by Quantity')

    histogram_boxplot(df, 'total_revolving_bal', 'Customer Total Revolving Balance')

    histogram_boxplot(df, 'avg_open_to_buy')

    histogram_boxplot(df, 'total_trans_ct')

    histogram_boxplot(df, 'avg_utilization_ratio')

    labeled_barplot(df, 'card_category', 'Customer Credit Card Types')

    labeled_barplot(df, 'marital_status', 'Customer Marital Status')

    labeled_barplot(df, 'attrition_flag', 'Customer Status')

    labeled_barplot(df, 'dependent_count', 'Customer Dependents by Quantity')

    print(df['dependent_count'].value_counts(1).sort_values(ascending=False))

    labeled_barplot(df, 'gender', 'Customers by gender')

    distribution_plot_wrt_target(df, 'total_trans_amt', 'attrition_flag')

    distribution_plot_wrt_target(df, 'total_trans_ct', 'attrition_flag')

    distribution_plot_wrt_target(df, 'total_revolving_bal', 'attrition_flag')

    distribution_plot_wrt_target(df, 'total_amt_chng_q4_q1', 'attrition_flag')

    distribution_plot_wrt_target(df, 'total_ct_chng_q4_q1', 'attrition_flag')

    distribution_plot_wrt_target(df, 'avg_utilization_ratio', 'attrition_flag')

    stacked_barplot(df, 'income_category', 'attrition_flag')

    stacked_barplot(df, 'card_category', 'attrition_flag')

    stacked_barplot(df, 'gender', 'attrition_flag')

    stacked_barplot(df, 'total_relationship_count', 'attrition_flag')

    stacked_barplot(df, 'education_level', 'attrition_flag')

    stacked_barplot(df, 'contacts_count_12_mon', 'attrition_flag')

    # Assume df is your DataFrame with numerical attributes
    df_numerical = df.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    correlation_matrix = df_numerical.corr()

    # Display the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
