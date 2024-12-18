import spacy
from Chapter01.dividing_into_sentences import read_text_file

text = read_text_file("sherlock_holmes_1.txt")
nlp = spacy.load("en_core_web_md")
doc = nlp(text)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.text)

    