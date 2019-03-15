import numpy as np
import pandas as pd
from scipy import stats
import math
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

import visualizer as viz

def get_predictions(models, X, y):
    validation_size = 0.20
    seed = 73
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
    models['LR'] = LogisticRegression()
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier(random_state=13)
    models['NB'] = GaussianNB()
    models['SVC'] = SVC(probability=True)
    models['XGB'] = XGBClassifier()
    
    return models