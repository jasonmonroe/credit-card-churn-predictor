# src/utils.py

import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
from src.config import *

# ==================================
#  HELPER FUNCTIONS
# ==================================

# --- Functions --- #
def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return 'RUN ID:' + str(random.randint(10000, 99999))


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    hours, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"Run Time: {get_time(start_time_int)}")


def show_banner(title: str, section: str='') -> None:
    padding = 2
    strlen = len(title) + padding

    # Top line
    #print("\n")
    print('# ', end='')
    print('=' * strlen)
    print('#', end='')

    # Show title
    print('  ' + title)

    print('# ', end='')
    print('=' * strlen)
    print('')
    #print('# ', end='')

    # Show section
    if section:
        print(' ' + section)
        print("\n")

# function to plot a boxplot and a histogram along the same scale.
def histogram_boxplot(data: pd.DataFrame, feature: str, chart_title: str='', figsize:tuple=(12, 7), kde:bool=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
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
        #plt.title(chart_title.title() + 'Histogram')

    plt.title(chart_title_str + ' Histogram')


    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram

    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# function to create labeled barplots
def labeled_barplot(chart_data: pd.DataFrame, feature: str, chart_title: str='', perc:bool=False, n=None):
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
    #plt.ylabel("Income Range")
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

    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter,
        ascending=False
    )

    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

### Function to plot distributions
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

# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target) -> pd.DataFrame:
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred) # to compute Accuracy
    recall = recall_score(target, pred) # to compute Recall
    precision = precision_score(target, pred) # to compute Precision
    f1 = f1_score(target, pred) # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1

        },
        index=[0],
    )

    return df_perf

def plot_confusion_matrix(model, X, y_true):
    """
    Generates a heatmap for the confusion matrix of a given model and dataset.

    Parameters:
    model: Trained model
    X: Feature data to make predictions
    y_true: True target labels

    Returns:
    Heatmap showing TP, FP, TN, FN.
    """

    # Predict the target for the given features
    y_pred = model.predict(X)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages for each cell in the confusion matrix
    cm_percentage = cm / cm.sum() * PERCENTILE

    # Add a label to chart.
    labels = np.asarray([
        [f"{int(cm[i, j])}\n{cm_percentage[i, j]:.2f}%" for j in range(len(cm))]
        for i in range(len(cm))
    ])

    # Display the confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    hm = sns.heatmap(cm, annot=labels, fmt='', cbar=False,
                xticklabels=model.classes_, yticklabels=model.classes_)


    plt.title("Confusion Matrix Heatmap")
    plt.show()

    # Extract TP, FP, TN, FN and print them
    TN, FP, FN, TP = cm.ravel()

    print(f"\nTrue Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
