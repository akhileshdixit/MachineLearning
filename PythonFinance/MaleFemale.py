# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 02:26:45 2017

@author: Akhilesh
"""

from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))


'''
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
#1
clf1 = svm.SVC()
#2
clf2 = GaussianNB()
#3
clf3 = neighbors.KNeighborsClassifier()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

#Support Vector classifier
clf1 = clf1.fit(X,Y)

#Naive Bayes
clf2 = clf2.fit(X,Y)

#K neighbors classifier
clf3 = clf3.fit(X,Y)
#test data

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

Y_prediction = clf.predict([[190, 70, 43]])

Y_prediction1 = clf1.predict(X_test)

Y_prediction2 = clf2.predict(X_test)

Y_prediction3 = clf3.predict(X_test)

#CHALLENGE compare their reusults and print the best one!

print("Prediction for Decision Tree Classifier : ",Y_prediction)

print("Prediction for Naive Bayes : ",Y_prediction2)

print("Accuracy for Naive Bayes : ",accuracy_score(Y_test,Y_prediction2))

print("Accuracy for SVM : ",accuracy_score(Y_test,Y_prediction1))

print("Accuracy for K neighbors : ",accuracy_score(Y_test,Y_prediction3))

'''


'''
from sklearn import tree

# [height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

print (prediction)
'''
