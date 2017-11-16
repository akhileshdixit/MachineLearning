# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:30:21 2017

@author: Akhilesh

Credit Card Fraud Analysis

"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def fraudAnalysis():
    masterData = pd.read_csv('FraudAnalysis.csv')
    #print(masterData.head())
    print(masterData.Fraudulent.value_counts())
    print(masterData.CardCountry.value_counts())
    
    encoded_countries = pd.get_dummies(masterData.CardCountry, prefix='cc_')
    print(encoded_countries.head())
    print(masterData.head())
    
    y = masterData.Fraudulent
    X = masterData[['Amount', 'CardUse24Hour', 'cc_AU', 'cc_GB']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    lr_model = LogisticRegression().fit(X_train, y_train)    
    print(lr_model.coef_)
    print(lr_model.intercept_)
    
    y_test_scores_lr = lr_model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_scores_lr)
    
    print(roc_auc_score(y_test, y_test_scores_lr))
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=20).fit(X_train, y_train)    
    
    y_test_scores_dt = [x[1] for x in dt_model.predict_proba(X_test)]
    print(roc_auc_score(y_test, y_test_scores_dt))
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=100).fit(X_train, y_train)
    
    y_test_scores_rf = [x[1] for x in rf_model.predict_proba(X_test)]
    
    print(roc_auc_score(y_test, y_test_scores_rf))
    
    
    

if __name__ == '__main__':
    fraudAnalysis()