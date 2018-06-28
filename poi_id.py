#!/usr/bin/python

import pickle
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 
                 'director_fees','to_messages','from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Create new features
employee_names = data_dict.keys()

for person in employee_names:
    
    messages_to_this_person = float(data_dict[person]['to_messages'])
    messages_from_this_person = float(data_dict[person]['from_messages'])
    messages_sent_to_poi = float(data_dict[person]['from_this_person_to_poi'])
    messages_received_from_poi = float(data_dict[person]['from_poi_to_this_person'])

    if messages_from_this_person > 0:
        data_dict[person]['fraction_sent_to_poi'] = messages_sent_to_poi / messages_from_this_person
    else:
        data_dict[person]['fraction_sent_to_poi'] = 0

    if messages_to_this_person > 0:
        data_dict[person]['fraction_received_from_poi'] = messages_received_from_poi / messages_to_this_person
    else:
        data_dict[person]['fraction_received_from_poi'] = 0

features_list.append('fraction_sent_to_poi')
features_list.append('fraction_received_from_poi')

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

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=.90, stratify=labels, random_state=42)

#features selected
nan_counts = {}
for feature in features_list:
    for k,v in my_dataset.iteritems():
	    if v[feature] != 'NaN':
		    nan_counts[feature]=nan_counts.get(feature,0)+1

number_of_features=10
selector = SelectKBest(k=number_of_features)
selector_transformed = selector.fit_transform(features_train,labels_train)
indices = selector.get_support(True)

features_selected_list = ['poi']
print "\n"
for index in indices:
    print '%-25s Score: %-15f Non-nan: %d' % (features_list[index + 1], selector.scores_[index], nan_counts[features_list[index + 1]])
    features_selected_list.append(features_list[index + 1])
print "\n"	
features_list = features_selected_list
#Construct Classifier

#redefine in term of updated features list
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Using stratified shuffle split cross validation because of the small size of the dataset
shuffle_split = StratifiedShuffleSplit(labels, 1000, random_state=55)

# Build pipeline
estimators = []
estimators.append(('minmax_scaler', MinMaxScaler()))

#choose method
estimators.append(('decision_tree', DecisionTreeClassifier()))
#estimators.append(('logistic_reg', LogisticRegression()))

model = Pipeline(estimators)

#parameters for logistic regression
parameters_lr = dict(logistic_reg__class_weight=['balanced'],
                      logistic_reg__solver=['liblinear'],
                      logistic_reg__C=range(1, 5),
                      logistic_reg__random_state=range(52,54))

#parameters for decision tree
parameters_dt = dict(decision_tree__min_samples_leaf=range(1, 5),
                     decision_tree__max_depth=range(1, 5),
                     decision_tree__class_weight=['balanced'],
                     decision_tree__criterion=['gini', 'entropy'])

					 
#Use GridSearchCV to test different classifiers and parameters
#I tested logistic_reg and decision_tree

cv = GridSearchCV(model, param_grid=parameters_dt, scoring='f1', cv=shuffle_split)

cv.fit(features, labels)
print "\n"
print cv.best_params_
print "\n"
clf=cv.best_estimator_

#clf = DecisionTreeClassifier(min_samples_leaf=2,
                     #max_depth=2,
                     #class_weight='balanced',
                     #criterion='entropy')

test_classifier(clf, my_dataset, features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)