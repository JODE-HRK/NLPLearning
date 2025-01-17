import nltk
import string
import csv
import json
import pandas as pd
import gensim
from langdetect import detect
import pickle
from nltk import FreqDist
from Chapter01.dividing_into_sentences import divide_into_sentences_nltk
from Chapter01.tokenization import tokenize_nltk
from Chapter01.removing_stopwords import read_in_csv

def get_yelp_reviews(filename):
    reader = pd.read_json(filename, orient="records", lines=True, chunksize=10000)
    chunk = next(reader)
    text = ''
    for index, row in chunk.iterrows():
        row_text = row['text']
        lang = detect(row_text)
        if (lang == "en"):
            text = text + row_text.lower()
    return text

def get_phrases(text):
    words = nltk.tokenize.word_tokenize(text)
    phrases = {}
    current_phrase = []
    for word in words:
        if (word in stopwords or word in string.punctuation):
            if(len(current_phrase) > 1):
                phrases[" ".join(current_phrase)] = "_".join(current_phrase)
                current_phrase = []
        else:
            current_phrase.append(word)

    if (len(current_phrase) > 1):
        phrases[" ".join(current_phrase)] = "_".join(current_phrase)
    
    return phrases

def replace_phrases(phrases_dict, text):
    for phrase in phrases_dict.keys():
        text = text.replace(phrase, phrases_dict[phrase])
    return text

def write_text_to_file(text, filename):
    text_file = open(filename, "w", encoding="utf-8")
    text_file.write(text)
    text_file.close()

def create_and_save_frequency_dist(word_list, filename):
    fdist = FreqDist(word_list)
    pickle.dump(fdist, open(filename, 'wb'))
    return fdist

def create_and_save_word2vec_model(words, word2vec_model_path):
    model = gensim.models.Word2Vec(words, window=5, vector_size=200)
    model.train(words, total_examples=len(words), epochs=200)

    pickle.dump(model, open(word2vec_model_path, 'wb'))
    return model

stopwords_file = "/home/kuan/Desktop/NLP/Chapter01/stopwords.csv"
stopwords = read_in_csv(stopwords_file)
yelp_reviews_file = "/home/kuan/Desktop/NLP/Chapter03/yelp_dataset/yelp_academic_dataset_review.json"

text = get_yelp_reviews(yelp_reviews_file)
phrases = get_phrases(text)
text = replace_phrases(phrases, text)
write_text_to_file(text, "/home/kuan/Desktop/NLP/Chapter03/all_text.txt")

sentences = divide_into_sentences_nltk(text)
all_sentence_words = [tokenize_nltk(sentence.lower()) for sentence in sentences]
flat_word_list = [word.lower() for sentence in all_sentence_words for word in sentence]
fdist = create_and_save_frequency_dist(flat_word_list, "Chapter03/fdist.bin")
print(fdist.most_common()[:1000])

model = create_and_save_word2vec_model(all_sentence_words, "/home/kuan/Desktop/NLP/Chapter03/phrases.model")

words = model.wv.most_similar("highly_recommend", topn=10)
print(words)
words = model.wv.most_similar("happy_hour", topn=10)
print(words)
words = model.wv.most_similar("fried_rice", topn=10)
print(words)
words = model.wv.most_similar("dim_sum", topn=10)
print(words)