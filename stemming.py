# stemming to remove -ing -ed ....

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

words = ['leaf', 'leaves', 'wrote', 'done', 'stemming', 'effective']

stemmed_words = [stemmer.stem(word) for word in words]

print(stemmed_words)