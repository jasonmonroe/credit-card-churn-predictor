# src/eda.py

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import PERCENTILE


def run_eda(df: pd.DataFrame):
    print('\n--- Running Exploratory Data Analysis Pipeline ---')

    # How is the total transaction amount distributed?
    trans_amt_dist = df['total_trans_amt'].describe().T

    print('Distribution of total transaction amount:')
    for key, value in trans_amt_dist.items():
        print(f"{key} is {value}")

    # What is the distribution of the level of education of customers?
    edu_level_dist = df['education_level'].value_counts()
    sum_val = edu_level_dist.sum()

    print('Distribution of education level')
    for key, value in edu_level_dist.items():
        pct = round((value / sum_val) * PERCENTILE)
        print(f"{key} is {value} or {pct}%")

    # Display box plot
    labeled_barplot(df, 'education_level', 'Customers by Education Level')

    # What is the distribution of the level of income of customers?

    income_dist = df['income_category'].value_counts()
    sum_val = income_dist.sum()

    print('Distribution of income level')

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

    print('Inactive Data')
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

    show_correlation_matrix(df)


def show_correlation_matrix(df: pd.DataFrame):
    # Assume df is your DataFrame with numerical attributes
    # Calculate the correlation matrix
    df_numerical = df.select_dtypes(include=['number'])
    correlation_matrix = df_numerical.corr()

    # Display the correlation matrix
    title = 'Correlation Matrix'
    plt.figure(num=f'{title}', figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()


# function to plot a boxplot and a histogram along the same scale.
def histogram_boxplot(data: pd.DataFrame, feature: str, chart_title: str = '', figsize: tuple = (12, 7),
                      kde: bool = False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of a figure (default (12,7))
    kde: whether to show a density curve (default False)
    bins: number of bins for histogram (default None)
    """

    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots

    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column

    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram

    # Set Histogram Title
    chart_title_str = feature.title().replace('_', ' ')
    plt.xlabel(chart_title_str)

    if chart_title:
        chart_title_str = chart_title.title().replace('_', ' ')

    plt.title(chart_title_str + ' Histogram')

    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram

    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# function to create labeled barplots
def labeled_barplot(chart_data: pd.DataFrame, feature: str, chart_title: str = '', perc: bool = False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(chart_data[feature])  # length of the column
    count = chart_data[feature].nunique()

    if n is None:
        plt.figure(figsize=(count + 1, 5))

    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=chart_data,
        x=feature,
        palette="Paired",
        order=chart_data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                PERCENTILE * (p.get_height() / total)
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    if chart_title:
        plt.title(chart_title)

    plt.xlabel(feature.title().replace('_', ' '))
    plt.show()  # show the plot


# Function to plot stacked bar chart
def stacked_barplot(data: pd.DataFrame, predictor: str, target: str):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter,
        ascending=False
    )

    print(tab1)
    print('\n')

    tab = pd.crosstab(data[predictor], data[target], normalize='index').sort_values(
        by=sorter,
        ascending=False
    )

    tab.plot(kind='bar', stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc='lower left',
        frameon=False,
    )

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


# Function to plot distributions
def distribution_plot_wrt_target(data, predictor, target):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model: Any, feature: pd.DataFrame, y_true: pd.Series, title: str = '') -> None:
    """
    Generates a heatmap for the confusion matrix of a given model and dataset.

    Parameters:
    model: Trained model
    feature: Feature data to make predictions
    y_true: True target labels
    title: Title for the plot

    Returns:
    Heatmap showing TP (True Positives), FP (False Positives), TN (True Negatives), FN (False Negatives).
    """

    # Predict the target for the given features
    y_pred = model.predict(feature)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages for each cell in the confusion matrix
    cm_percentage = cm / cm.sum() * PERCENTILE

    # Add a label to the chart.
    labels = np.asarray([
        [f"{int(cm[i, j])}\n{cm_percentage[i, j]:.2f}%" for j in range(len(cm))]
        for i in range(len(cm))
    ])

    # Display the confusion matrix as a heatmap
    title = f'{title} Plot Confusion Matrix'.strip()
    plt.figure(num=f'{title}', figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=labels,
        fmt='',
        cbar=False,
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.title(title)
    plt.show()

    # Extract TP, FP, TN, FN and print them
    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

    print(f"👍🏾➕\tTrue Positives (TP): {true_positives}")
    print(f"👎🏾➕\tFalse Positives (FP): {false_positives}")
    print(f"👍🏾➖\tTrue Negatives (TN): {true_negatives}")
    print(f"👎🏾➖\tFalse Negatives (FN): {false_negatives}")
