def classify(features_train, labels_train):
	"""return a trained decision tree classifer"""
    
    ### import decision tree from sklearn
    from sklearn import tree
    
    ### Create classifier and train it
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)

    return clf