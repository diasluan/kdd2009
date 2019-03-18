import numpy as np
import pandas as pd
import pickle
from scipy import stats
import math
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
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

import xgboost as xgb
from xgboost import XGBClassifier

#sklean model selection http://scikit-learn.org/
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

def feature_importance(X, y, threshold=0.005):
    estimator = GradientBoostingClassifier(
        max_depth=3,
        subsample=0.8,
        verbose=1,  
        random_state=73
    )
    estimator.fit(X, np.array(y).ravel())
    print('|-------|')

    select = SelectFromModel(estimator, threshold=threshold, prefit=True)
    return select.transform(X)
                  
def split_dataset(features, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels, random_state=73)
    y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
    
    return X_train, X_test, y_train, y_test
                  
def train_and_report(models, X, y):
    results = []
    for name in models.keys():
        model = models[name]
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print("AUC: %.3f (+/- %.3f) [%s]" %(scores.mean(), scores.std(), name))
                  
def gbc_params_optimizer(X_train, y_train, n_estimators, learning_rate, min_samples_split, min_samples_leaf, max_depth, max_features, subsample, params, cv=5):
    np.random.seed(0)

    model = GradientBoostingClassifier(n_estimators=n_estimators, 
                                       learning_rate=learning_rate,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, 
                                       max_depth=max_depth, 
                                       max_features=max_features, 
                                       subsample=subsample, 
                                       random_state=0)

    
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=params, 
                               scoring='roc_auc', 
                               n_jobs=-1, 
                               iid=False, 
                               cv=cv)

    grid_search.fit(X_train, y_train)

    results = grid_search.cv_results_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(best_params, best_score)
    
    return model, best_params, best_score
                  
def gbc_lr_optimizer(X_train, y_train, n_estimators, learning_rate, min_samples_split, min_samples_leaf, 
                     max_depth, max_features, subsample, params, cv=5, dividers=[2,5,2,5]):
    models = np.array([])
    scores = np.array([])
    
    for div in dividers:
        np.random.seed(0)
        
        learning_rate /= div
        n_estimators *= div
        
        model = GradientBoostingClassifier(n_estimators = n_estimators, 
                                         learning_rate = learning_rate,
                                         min_samples_split = min_samples_split,
                                         min_samples_leaf = min_samples_leaf, 
                                         max_depth = max_depth, 
                                         max_features = max_features, 
                                         subsample = subsample, 
                                         random_state = 0)

        cv_scores = cross_val_score(model, X_train, y_train, scoring = 'roc_auc', cv=cv, n_jobs=1)
        scores = np.append(scores, cv_scores.mean())
        
        print('n_estimators: {} | learning_rate: {} | score: {}'.format(n_estimators, learning_rate, cv_scores.mean()))
        
        models = np.append(models, model)

    return models, scores
                  
def save_model(model, title, path='./models/'):
    filename = path + title
    pickle.dump(model, open(filename, 'wb'))