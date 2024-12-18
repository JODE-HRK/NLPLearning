# import nltk
# filename = "sherlock_holmes_1.txt"
# file = open(filename, "r", encoding="utf-8")
# text = file.read()
# text = text.replace("\n", " ")
# tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
# sentences = tokenizer.tokenize(text)

# print("nltk version:")
# print(sentences)

import spacy
sfilename = "sherlock_holmes_1.txt"
sfile = open(sfilename, "r", encoding="utf-8")
stext = sfile.read()
stext = stext.replace("\n", " ")
nlp = spacy.load("en_core_web_sm")
doc = nlp(stext)
ssentences = [ssentence.text for ssentence in doc.sents]

print("spacy version:") 
print(ssentences)