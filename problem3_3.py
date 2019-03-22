import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import accuracy_score
import sys
import csv


class Classification:

    def __init__(self, inputfile, outputfile):
        self.inputFile = inputfile
        self.outputFile = outputfile
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None

    def dataPrep(self):
        data = pd.read_csv(self.inputFile)
        features = data[["A", "B"]].values
        label = data['label'].values
        x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=42)
        self.Xtrain = x_train
        self.ytrain = y_train
        self.Xtest = x_test
        self.ytest = y_test
        return self.Xtrain, self.ytrain, self.Xtest, self.ytest

    def writeOutput(self, clf_name, train_score, test_score):
        row = [clf_name, train_score, test_score]
        with open(self.outputFile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            writer.writerow(row)



    def svmLinear(self):
        params = {
            'C': [0.1, 0.5, 1, 5, 10, 50, 100],
            'kernel': ['linear']
        }
        self.dataPrep()
        clf = SVC()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv = 5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='svm_linear', train_score=train_score, test_score=test_score)

    def svmPoly(self):
        params = {
            'C': [0.1, 1, 3],
            'degree': [4, 5, 6],
            'gamma': [0.1, 0.5],
            'kernel': ['poly']
        }
        self.dataPrep()
        clf = SVC()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv=5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='svm_polynomial', train_score=train_score, test_score=test_score)

    def svmRBF(self):
        params = {
            'C': [0.1, 0.5, 1, 5, 10, 50, 100],
            'gamma': [0.1, 0.5, 1, 3, 6, 10],
            'kernel': ['rbf']
        }
        self.dataPrep()
        clf = SVC()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv=5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='svm_rbf', train_score=train_score, test_score=test_score)

    def logisticReg(self):
        params = {
            'C': [0.1, 0.5, 1, 5, 10, 50, 100],
            'solver': ['liblinear']
        }
        self.dataPrep()
        clf = LogisticRegression()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv=5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='logistic', train_score=train_score, test_score=test_score)

    def knnClass(self):
        params = {
            'n_neighbors': range(1, 51),
            'leaf_size': range(5, 65, 5),
            'algorithm': ['auto']
        }
        self.dataPrep()
        clf = KNeighborsClassifier()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv=5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='knn', train_score=train_score, test_score=test_score)

    def decisionTree(self):
        params = {
            'max_depth': range(1, 51),
            'min_samples_split': range(2, 11)
        }
        self.dataPrep()
        clf = DecisionTreeClassifier()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv=5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='decision_tree', train_score=train_score, test_score=test_score)

    def randomForest(self):
        params = {
            'max_depth': range(1, 51),
            'min_samples_split': range(2, 11)
        }
        self.dataPrep()
        clf = RandomForestClassifier()
        clf = GridSearchCV(clf, params, n_jobs=-1, cv=5)
        clf.fit(self.Xtrain, self.ytrain)
        train_score = clf.best_score_
        test_score = clf.score(self.Xtest, self.ytest)
        self.writeOutput(clf_name='random_forest', train_score=train_score, test_score=test_score)


if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    cls = Classification(inputfile=inputfile, outputfile=outputfile)
    cls.svmLinear()
    cls.svmPoly()
    cls.svmRBF()
    cls.logisticReg()
    cls.knnClass()
    cls.decisionTree()
    cls.randomForest()



