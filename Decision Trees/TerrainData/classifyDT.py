import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## DECISION TREE #################################

# imports
from sklearn import tree
from sklearn.metrics import accuracy_score

# creating classifier and training it
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

# using classifier to make prediction on test data
pred = clf.predict(features_test)

# computing accuracy of label data
acc = accuracy_score(pred,labels_test)

def submitAccuracies():
  return {"acc":round(acc,3)}