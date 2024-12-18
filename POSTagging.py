# import spacy
# filename = "sherlock_holmes_1.txt"
# file = open(filename, "r", encoding="utf-8")
# text = file.read()
# text = text.replace("\n", " ")
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(text)
# words = [token.text for token in doc]
# pos = [token.pos_ for token in doc]
# word_pos_tuples = list(zip(words, pos))
# print(word_pos_tuples)

import nltk
# nltk.download('tagsets')
# nltk.help.upenn_tagset()
filename = "sherlock_holmes_1.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()
text = text.replace("\n", " ")
words = nltk.tokenize.word_tokenize(text)
words_with_pos = nltk.pos_tag(words)
print(words_with_pos)