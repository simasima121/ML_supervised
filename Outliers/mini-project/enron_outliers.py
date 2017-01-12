#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("final_project_dataset.pkl", "r") )

### Removing outlier from dictionary 
i = 0
popped = []
for d in data_dict:
	if data_dict[d]["total_payments"] > 300000000 and data_dict[d]["total_payments"] != "NaN":
		popped.append([d,i])
	i+=1
data_dict.pop(popped[0][0], popped[0][1])

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


