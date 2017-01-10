#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)
print len(enron_data["HAYES ROBERT E"])

### Find out how many Persons of Interest are in the dataset
count = 0
for e in enron_data:
	if enron_data[e]["poi"]:
		count += 1	
print count

### How many POIs in total
pois = open("../final_project/poi_names.txt", "r")
count = 0
for p in pois:
	if p[0] == "(":
		count += 1
print count

print enron_data["PRENTICE JAMES"]
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

### Who took home the most money out of Lay, Skilling and Fastow
for e in enron_data:
	if "SKILL" in e or "LAY" in e or "FASTOW" in e:
		print e
lay = enron_data["LAY KENNETH L"]
skilling = enron_data["SKILLING JEFFREY K"]
fastow = enron_data["FASTOW ANDREW S"]

print lay["total_payments"]
print skilling["total_payments"]
print fastow["total_payments"]

### How many people have a quantified salary or known email address
count = 0
counted = 0
for e in enron_data:
	if enron_data[e]['salary'] != "NaN":
		count +=1 
	if enron_data[e]['email_address'] != "NaN":
		counted +=1 
print count, counted

### Percentage of people with NaN for total_payments
count = 0
for e in enron_data:
	if enron_data[e]['total_payments'] == "NaN":
		count += 1.0
print (count/len(enron_data)) * 100.0

### Percentage of POIs with NaN for total_payments
count = 0
for e in enron_data:
	if enron_data[e]['total_payments'] == "NaN" and enron_data[e]['poi'] == True:
		print enron_data[e]
		count += 1.0
print (count/len(enron_data)) * 100.0