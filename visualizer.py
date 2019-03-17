from jupyterthemes import jtplot
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import missingno as msno
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import math

jtplot.style(ticks=True, figsize=(15, 10))

def highlight_error(s):
    is_error = s != s.iloc[0]
    return ['color: red' if v else 'color: black' for v in is_error]

def decision_boundaries(models, X, y, cols=2):        
    fig = plt.figure()
    rows = math.ceil(len(models) / (cols * 1.0))
    gs = gridspec.GridSpec(rows, cols)
    grid = []
    
    for r in range(rows):
        for c in range(cols):
            grid.append((r,c))

    clf_list = models.values()
    labels = models.keys()
    
    for clf, label, grd in zip(clf_list, labels, grid):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(label)

    plt.show()

def class_plot(df):
    sns.pairplot(df, hue="class", diag_kind="kde")
    plt.show()
    
def models_correlation(df):
    corr = df.iloc[:, 1:].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, mask=mask, cmap="YlGnBu")
    plt.show()
    
def plot_missing_matrix(df):
    missing_data_df = df.columns[df.isnull().any()].tolist()
    msno.matrix(df[missing_data_df], sparkline=False, fontsize=12)
    plt.show()


def plot_missing_bar(df):
    missing_data_df = df.columns[df.isnull().any()].tolist()
    msno.bar(df[missing_data_df], log=False, figsize=(30, 18))
    plt.show()


def plot_missing_heatmap(df):
    missing_data_df = df.columns[df.isnull().any()].tolist()
    msno.heatmap(df[missing_data_df], figsize=(20, 20))
    plt.show()

def plot_categories_per_feature(dataset): 
    count = dataset.apply(lambda x: len(set(x)))
    print(count)
    plt.hist(count)
    