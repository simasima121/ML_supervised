#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
	
import sys
from time import time
sys.path.append("tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### Create the classifier
clf = SVC(C=10000, kernel="rbf")

### Slice training dataset down to 1% of it's original size0
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0,3), "s"

### use trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0,3), "s"

### calculate and return the accuracy on the test data
accuracy = accuracy_score(pred, labels_test)
print accuracy

### Prediction for element 10, 26, 50 of the test set
elem10 = pred[10]
elem26 = pred[26]
elem50 = pred[50]

print elem10, elem26, elem50

### Number of elements in test set predicted to be Chris
count = 0

for i in pred:
	if i == 1:
		count +=1
print count

### Alternate way to find number of elements predicted to be Chris
print len(filter(lambda x:x==1,pred))
