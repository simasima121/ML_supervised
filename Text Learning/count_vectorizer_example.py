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

## You get a bag of tuples and integer values, zero-indexed
## Way to intepret is for (1, 7) 1, document 1/word number 7 - occurs 1 time
## Words are all ordered in bag_of_words, thus for (2,15) it means the 15th word in bag_of_words, not in str3
print bag_of_words

## Checking if (1, 6) 3 is actually the word great - it is!
print vectoriser.vocabulary_.get("great")