## See following link for other stemmers http://www.nltk.org/howto/stem.html

from nltk.stem.snowball import SnowballStemmer

## Have to declare a language I want to stem with
stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")
print stemmer.stem("responsivity")
print stemmer.stem("unresponsive")


