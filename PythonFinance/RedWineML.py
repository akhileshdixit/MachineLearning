# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:59:03 2017

@author: Akhilesh
"""

# Import libraries and modules
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# Load red wine data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url)

print (data.head())

data = pd.read_csv(dataset_url, sep=';')

print (data.head())

print (data.shape)
print (data.describe())

# Split data into training and test sets.

y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# Declare data preprocessing steps

# Fitting the Transformer API
scaler = preprocessing.StandardScaler().fit(X_train)

# Applying transformer to training data

X_train_scaled = scaler.transform(X_train)

print(X_train_scaled.mean(axis=0))

print(X_train_scaled.std(axis=0))

# Applying transformer to test data
X_test_scaled = scaler.transform(X_test)

print (X_test_scaled.mean(axis=0))

print (X_test_scaled.std(axis=0))

# Pipeline with preprocessing and model

pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# Declare hyperparameters to tune

hyperparameters = {'randomforestregressor__max_features':['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth':[None, 5, 3, 1]}

# Tune model using cross-validation pipeline

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)

# Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)

# Evaluate model pipeline on test data

pred = clf.predict(X_test)
print (r2_score(y_test, pred))
print (mean_squared_error(y_test, pred))


# Save model for future use

joblib.dump(clf, 'rf_regressor.pkl')

