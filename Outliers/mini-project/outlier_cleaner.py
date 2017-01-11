#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        predictions is a list of predicted targets that come from your regression
        ages is the list of ages in the training set
        net_worths is the actual value of the net worths in the training set

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### residual error is actual - predicted
    re = abs(net_worths - predictions)

    ### zip allows data to be stored in tuple linked by the ith value
    cleaned_data = zip(ages, net_worths, re)

    ### sort cleaned_data 
    cleaned_data = sorted(cleaned_data, key=lambda cleaned_data: cleaned_data[2])
    
    ### clean away the 10% of points with the largest re
    cleaned_data = cleaned_data[:int(0.9*len(cleaned_data))]

    return cleaned_data

