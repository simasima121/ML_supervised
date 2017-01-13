from nltk.corpus import stopwords

## Have to declare a language I want my stopwords in
sw = stopwords.words("english")
print sw[0], sw[10]
print len(sw)