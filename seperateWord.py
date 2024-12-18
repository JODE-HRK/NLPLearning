# import nltk
# filename = "sherlock_holmes_1.txt"
# file = open(filename, "r", encoding="utf-8")
# text = file.read()
# text = text.replace("\n", " ")

# words = nltk.tokenize.word_tokenize(text)
# print(words)

import spacy
filename = "sherlock_holmes_1.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()
text = text.replace("\n", " ")
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
words = [token.text for token in doc]
print(words)