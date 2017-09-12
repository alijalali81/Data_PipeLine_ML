# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:03:35 2017

@author: ajalali
"""

from sklearn import datasets

iris=datasets.load_iris()
x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

from sklearn import tree
my_clf=tree.DecisionTreeClassifier()


from sklearn.neighbors import KNeighborsClassifier
my_clf=KNeighborsClassifier()

my_clf.fit(x_train,y_train)
y_predict=my_clf.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predict)

from sklearn.model_selection import learning_curve
#
#from sklearn.model_selection import validation_curve
#
train_sizes, train_scores, valid_scores = learning_curve(
my_clf, x, y, train_sizes=[0.5, 0.2,0.3], cv=2)