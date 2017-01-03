import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## DECISION TREE #################################

### imports
from sklearn import tree
from sklearn.metrics import accuracy_score

### create classifier and train it
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

### use classifier to make prediction on test data
pred = clf.predict(features_test)

### compute accuracy on label data
acc = accuracy_score(pred,labels_test)

def submitAccuracies():
  return {"acc":round(acc,3)}

print submitAccuracies()