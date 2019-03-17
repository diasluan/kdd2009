import numpy as np
import pandas as pd
from scipy import stats
import math
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

# libs basicas data science
from sklearn import datasets
import numpy as np
import pandas as pd
from scipy import stats
import math

#libs visualizacao
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import Image
from IPython.core.display import HTML
from mlxtend.plotting import plot_decision_regions

#sklean model selection http://scikit-learn.org/
from sklearn.model_selection import cross_val_score, train_test_split

#sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

import eli5
from eli5.sklearn import PermutationImportance

import visualizer as viz

def get_predictions(models, X, y, validation_size=0.20, seed=73):
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X, y, test_size=validation_size, random_state=seed)    
    predictions = [y_validation]
    labels = []
    for name in models.keys():
        model = models[name]
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_validation))
        labels.append(name)
    predictions_df = pd.DataFrame(data=np.transpose(predictions), columns=(['Y'] + list(models.keys()))) 
    return predictions_df

def print_predictions(models, X, y):
    predictions = get_predictions(models, X, y)
    return predictions.style.apply(viz.highlight_error, axis=1)


def train_and_report(models, X, y):
    results = []
    for name in models.keys():
        model = models[name]
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        print("Accuracy: %.3f (+/- %.3f) [%s]" %(scores.mean(), scores.std(), name))

def drop_min_unique_features(dataset, threshold):
    for col in dataset:
        if len(dataset[col].unique()) <= threshold: dataset.drop(col, inplace=True, axis=1)
    return dataset

def drop_max_unique_features(dataset, threshold):
    for col in dataset:
        if len(dataset[col].unique()) >= threshold: dataset.drop(col, inplace=True, axis=1)
    return dataset

def drop_max_null_features(dataset, threshold):
    for col in dataset:
        if sum(dataset[col].isnull()) >= threshold: dataset.drop(col, inplace=True, axis=1)
    return dataset

def get_models():
    models = {}
    models['LR'] = LogisticRegression(solver='lbfgs')
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier(random_state=73)
    models['NB'] = GaussianNB()
    models['SVC'] = SVC(probability=True)
    models['XGB'] = XGBClassifier(objective='binary:logistic', seed=73)
    models['RFC'] = RandomForestClassifier(random_state=73)
    models['GBC'] = GradientBoostingClassifier(random_state=73)
    
    return models

def get_summed_importances(variables, importances, plot=True):
    # Order importances from highest to lowest
    i = np.argsort(importances)[::-1]
    importances = importances[i]
    variables = variables[i]
    
    summed_importances = pd.DataFrame()
    for i in range(importances.size):
        importance_sum = importances[:(i+1)].sum()
        current_var = pd.Series([variables[i], importance_sum])
        summed_importances = summed_importances.append(current_var, ignore_index=True)

    if plot: plt.scatter(summed_importances.index, summed_importances.iloc[:,1])
    return summed_importances

def remove_unimportant_vars(features, summed_importances, threshold=0.99):
    important_vars = list(summed_importances[summed_importances.iloc[:,1] <= threshold].iloc[:,0])
    features = features.loc[:, important_vars]
    
    return features

def permutation_importance(estimator, X, y):
    permutation = PermutationImportance(estimator, random_state=73, prefit=True).fit(X, y)
    eli5.show_weights(permutation, feature_names=X.columns.tolist())
    
    select = SelectFromModel(permutation, threshold=0.05, prefit=True)
    return select.transform(X)