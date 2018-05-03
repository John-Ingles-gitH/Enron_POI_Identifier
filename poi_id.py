#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ["poi",'salary', 'deferral_payments', 'total_payments', 'bonus', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
                 'long_term_incentive', 'restricted_stock','to_messages','from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def train_and_test(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    true_negatives_train = 0
    false_negatives_train = 0
    true_positives_train = 0
    false_positives_train = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        training_predicitons = clf.predict(features_train)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
        for prediction, truth in zip(training_predicitons, labels_train):
            if prediction == 0 and truth == 0:
                true_negatives_train += 1
            elif prediction == 0 and truth == 1:
                false_negatives_train += 1
            elif prediction == 1 and truth == 0:
                false_positives_train += 1
            elif prediction == 1 and truth == 1:
                true_positives_train += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

    try:
        total_predictions_train = true_negatives_train + false_negatives_train + false_positives_train + true_positives_train
        accuracy_train = 1.0*(true_positives_train + true_negatives_train)/total_predictions_train
        precision_train = 1.0*true_positives_train/(true_positives_train+false_positives_train)
        recall_train = 1.0*true_positives_train/(true_positives_train+false_negatives_train)
        f1_train = 2.0 * true_positives_train/(2*true_positives_train + false_positives_train+false_negatives_train)
        f2_train = (1+2.0*2.0) * precision_train*recall_train/(4*precision_train + recall_train)
        print PERF_FORMAT_STRING.format(accuracy_train, precision_train, recall_train, f1_train, f2_train, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions_train, true_positives_train, false_positives_train, false_negatives_train, true_negatives_train)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."			 

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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

#create training/testing split
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=1)

#features selected
selector = SelectKBest(f_classif, k=3)
selector.fit(features,labels)
print selector.scores_
#selector.fit(features_train, labels_train)

#Drop features not selected from features_list
features_list_no_poi = features_list[1:]
features_list = [i for (i, v) in zip(features_list_no_poi, selector.get_support()) if v]
features_list.insert(0, "poi")
print features_list
#transformed
#features_train = selector.transform(features_train)
#features_test = selector.transform(features_test)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
scoring = {'Accuracy':make_scorer(accuracy_score),'Precision':make_scorer(precision_score),'Recall':make_scorer(recall_score)}
#parameters = {'max_depth':[1,2,3,4,5],'min_samples_split':[2,3,4,5]}
#dt = tree.DecisionTreeClassifier(random_state=1)
#clf = GridSearchCV(dt, parameters,scoring=scoring,refit='Accuracy')
#clf = GaussianNB()
  
#Decision Tree used as classifier
clf=tree.DecisionTreeClassifier(random_state=1)
train_and_test(clf,my_dataset,features_list)
#Fit to training data
#clf.fit(features_train,labels_train)
#print(clf.best_params_)
#predictions made using test data
#pred = clf.predict(features_test)

#accuracy, precision, and recall calculated
#acc = accuracy_score(pred, labels_test)
#prec = precision_score(pred, labels_test)
#recall = recall_score(pred, labels_test)

#print acc
#print prec
#print recall

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