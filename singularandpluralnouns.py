import nltk
from nltk.stem import WordNetLemmatizer
import inflect
from Chapter01.pos_tagging import pos_tag_nltk

filename = "sherlock_holmes.txt"
file = open(filename, "r", encoding="utf-8")
sherlock_holmes_text = file.read()
sherlock_holmes_text = sherlock_holmes_text.replace("\n", " ")
words_with_pos = pos_tag_nltk(sherlock_holmes_text)

def get_nouns(words_with_pos):
    noun_set = ["NN", "NNS"]
    nouns = [word for word in words_with_pos if word[1] in noun_set]
    return nouns

nouns = get_nouns(words_with_pos)
print(nouns)