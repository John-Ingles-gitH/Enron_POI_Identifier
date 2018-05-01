#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ["poi",'salary', 'deferral_payments', 'total_payments', 'bonus', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
                 'long_term_incentive', 'restricted_stock']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score

#create training/testing split
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=1)

#features selected
selector = SelectKBest(f_classif, k=4)
selector.fit(features,labels)
#selector.fit(features_train, labels_train)

#Drop features not selected from features_list
features_list_no_poi = features_list[1:]
features_list = [i for (i, v) in zip(features_list_no_poi, selector.get_support()) if v]
features_list.insert(0, "poi")
#transformed
#features_train = selector.transform(features_train)
#features_test = selector.transform(features_test)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
scoring = {'Accuracy':make_scorer(accuracy_score),'Precision':make_scorer(precision_score),'Recall':make_scorer(recall_score)}
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[1, 5]}
dt = tree.DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(dt, parameters,scoring=scoring, refit="Recall")
  
#Decision Tree used as classifier
#clf=tree.DecisionTreeClassifier(random_state=1)

#Fit to training data
#clf.fit(features_train,labels_train)

#predictions made using test data
#pred = clf.predict(features_test)

#accuracy, precision, and recall calculated
#acc = accuracy_score(pred, labels_test)
#prec = precision_score(pred, labels_test)
#recall = recall_score(pred, labels_test)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)