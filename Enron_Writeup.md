#Enron Machine Learning Project
###John Ingles

##Question 1

The goal of this project was to use machine learning techniques to create a model
that would predict whether or not someone was a 'person of interest' using data
compiled from the Enron Email Corpus.  This data set contains 146 people with 21
features for each person.

The features are:

####Financial

 * salary
 * deferral_payments
 * total_payments
 * loan_advances
 * bonus
 * restricted_stock_deferred
 * deferred_income
 * total_stock_value
 * expenses
 * exercised_stock_options
 * other
 * long_term_incentive
 * restricted_stock
 * director_fees

####Email
 
 * to_messages
 * email_address
 * from_poi_to_this_person
 * from_messages
 * from_this_person_to_poi
 * shared_receipt_with_poi

####POI Label
 
 * poi (Boolean, represented as integer)
 
In this data set, a person is noted as a 'person of interest' if they were later
indicted, reached a settlement or plea deal with the government, or testified in
exchange for immunity in regards to the Enron investigation.

It stands to reason that we could find trends in these peoples' financial and
email data that could be used to test for suspects of fraud in other places.
Machine learning is useful for this purpose because of its ability to learn from
information given to it and develop models to apply to new data sets.

A preliminary look at the data set identified several names that weren't people.

 * TOTAL
 * THE TRAVEL AGENCY IN THE PARK
 
These values were dropped since they clearly weren't employees.

##Question 2

The first step to building a machine learning model (after cleaning the data) is
to determine which features will be included and how many.  For this project, I 
used the univariate feature selection routine `SelectKBest` from `scikit-learn`.
I chose to select the 10 best scoring features from a split with 10% testing data.
I chose this amount because the number of people in the data set was over a
hundred, so any amount of features below 20 would be unlikely to overfit the
model.  I also started to lose predicative power after 11 features.  As for the 
classification test, I used `f_classif` which is an ANOVA test because the
classification of 'poi' is nominal and the features contain negative values. I
didn't use scaling when selecting the features because I was worried that the
importance of the larger values of some features would be reduced by the low
values of some features.

Before running the selection, I also created two new features:
 
 * fraction_sent_to_poi
 * fraction_received_from_poi

I created these features to track the email correspondence between persons of
interest and other employees.  The idea was that those who interacted the most 
with 'poi's could be 'poi's themselves.  They were created by calculating the
ratios:

 * fraction_sent_to_poi = 'from_this_person_to_poi' / 'from_messages'
 * fraction_received_from_poi = 'from_poi_to_this_person' / 'to_messages'

After running the selection, the 10 best features with their scores were:

 * exercised_stock_options - 25.99
 * total_stock_value - 24.93
 * bonus - 19.85
 * salary - 16.31
 * fraction_sent_to_poi - 12.68
 * deferred_income - 11.48
 * long_term_incentive - 10.78
 * expenses - 9.62
 * restricted_stock - 8.98
 * shared_receipt_with_poi - 8.84

##Question 3

After selecting the features, I modeled the data with `LogisticRegression` and
`DecisionTreeClassifier`.  The performance of each classifier when using the
testing script was:

Algorithm | Accuracy | Precision | Recall | StratifiedShuffleSplit Folds
--- | --- | --- | --- | ---
Logistic Regression | 0.76300 | 0.30401 | 0.60300 | 1000
Decision Tree | 0.86300 | 0.48995 | 0.67000 | 1000

Since it had higher scores for all three tests, I ending up using the
`DecisionTreeClassifier` even though it took much longer to run.

##Question 4

Tuning the parameters of an algorithm is the process of optimizing it's performance
with respect to a specific data set.  If no tuning is done, you risk modeling an
incomplete representation of the data. This tuning can be done by trail and error,
or by various methods that automate the process.  I chose to automate the process
using `GridSearchCV`.  My tested the `DecisionTreeClassifier` parameters:

 * min_samples_leaf - 1 to 5
 * max_depth - 1 to 5
 * criterion - 'gini' and 'entropy'

The best model as found by `GridSearchCV` was:

 * min_samples_leaf - 2
 * max_depth - 2
 * criterion - 'entropy'

##Question 5

Validation is when you check the predictive power of your model using data that
wasn't used to train it.  If you try to test your model using the data that you
used to train it, it will show an artificially high predictive power because it is
already well tuned to this data.  This is a classic example of overfitting your
model.  To get an accurate view of your model's predictive power, it is essential
that you test it with data it hasn't seen before.

In my analysis, I validated my model using `StratifiedShuffleSplit`.
`StratifiedShuffleSplit` takes the data and divides it into stratified randomized
folds that preserve the percentage of samples of each class ('poi': 0,1).

This was useful because the small size of the data set meant that many train/test
splits were likely to have zero 'poi' in them.  With `StratifiedShuffleSplit`,
I could test the performance of 1000 different train/test splits and keep the one
that performed best.

##Question 6

The final `DecisionTreeClassifier` model was tested using the testing script
provided by Udacity.  The metrics evaluated were accuracy, precision, and recall,
and F1 scores.  The test performed 15,000 predictions and found:

 * Accuracy: 0.86300 - 86.3% of people were correctly identified as 'poi' or 'non-poi'
 * Precision: 0.48995 - 49.0% of people identified as 'poi' were actually 'poi'
 * Recall: 0.67000 - 67.0% of 'poi's were correctly identified as 'poi'