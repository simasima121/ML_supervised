from sklearn.feature_extraction.text import CountVectorizer

## Create vectoriser
vectoriser = CountVectorizer()

## Creating test strings
str1 = "Hi Katie the self driving car will be late Best Sebastian"
str2 = "Hi Sebastian the machine learning class will be great great great Best Katie"
str3 = "Hi Katie the machine learning class will be most excellent"

## Putting them into list
email_list = [str1, str2, str3]

## fitting and transforming data for the vectoriser
bag_of_words = vectoriser.fit_transform(email_list)

## You get a bag of tuples and integer values
print bag_of_words