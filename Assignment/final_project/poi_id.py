#!/Users/sim/anaconda3/envs/enron/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np
import pandas as pd

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
## Loan advances removed as few entries, deferral_payments removed as gave worse fit
## Director Fees removed as only Non-POI entries, Restricted stock deferred removed as not enough entries for POI
features_list = ['poi', "bonus", "salary", "expenses", "deferred_income", "exercised_stock_options",  "long_term_incentive", "other", "restricted_stock", "restricted_stock_deferred", "total_payments", "total_stock_value", "from_poi_to_this_person", "from_this_person_to_poi"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### To spot outliers, I used link http://bl.ocks.org/dmenin/raw/d12a22521ad32cacc906/
### to identify potential issues and created my own boxplots with the following
### code.
def draw_boxplots(entry):
	data_dict.pop('TOTAL', 0)
	dat_dict_pd = [{k: data_dict[p][k]if data_dict[p][k] != 'NaN' else None for k in data_dict[p].keys()}for p in data_dict.keys()]
	dat_pd = pd.DataFrame(dat_dict_pd)

	# name your plot
	ax = dat_pd.boxplot(entry, by = 'poi')
	# save your plot to the folder that your code is in:
	fig = ax.get_figure()
	fig.savefig('%s.png' % entry)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing.
### Data is a numpy array
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Using Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, accuracy_score # measures of performance

clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)

print 'recall NB', recall_score(labels_test, pred1)
print 'precision ', precision_score(labels_test, pred1)
print 'accuracy_score ', accuracy_score(labels_test, pred1)
print "---------------------------------"

## RF Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'recall RF', recall_score(labels_test, pred)
print 'precision ', precision_score(labels_test, pred)
print 'accuracy_score ', accuracy_score(labels_test, pred)
print "---------------------------------"

## AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'recall AB', recall_score(labels_test, pred)
print 'precision ', precision_score(labels_test, pred)
print 'accuracy_score ', accuracy_score(labels_test, pred)
print "---------------------------------"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Trying DTC with AdaBoost with GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2, 10, 25, 50, 100, 150, 200]
             }
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
ABC = AdaBoostClassifier(base_estimator = DTC)
# run grid search
clf = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'recall Adaboost with GridSearchCV', recall_score(labels_test, pred)
print 'precision ', precision_score(labels_test, pred)
print 'accuracy_score ', accuracy_score(labels_test, pred)
print "---------------------------------"

### Using AdaBoost DT with PCA classifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, precision_score, accuracy_score # measures of performance

## creating a pipeline and fitting it
estimators = [('reduce_dim', PCA()), ('clf', AdaBoostClassifier(n_estimators=100))]
pipe = Pipeline(estimators)
clf = pipe
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'recall AdaBoost with PCA', recall_score(labels_test, pred)
print 'precision ', precision_score(labels_test, pred)
print 'accuracy_score ', accuracy_score(labels_test, pred)
print "---------------------------------"



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)