import gensim
import pickle
from os import listdir
from os.path import isfile, join
from Chapter03.bag_of_words import get_sentences
from Chapter01.tokenization import tokenize_nltk
from rtf_converter import rtf_to_txt

word2vec_model_path = "word2vec.model"
books_dir = "./archive"

def get_all_book_sentences(directory):
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and ".rtf" in f]
    # print(len(text_files))
    text_files = []
    for file in files:
        # rtf_content = file.read()
        plain_text = rtf_to_txt(file)
        text_files.append(plain_text)

    all_sentences = []
    for text_file in text_files:
        sentences = get_sentences(text_file)
        all_sentences = all_sentences + sentences
    return all_sentences

def train_word2vec(words, word2vec_model_path):
    model = gensim.models.Word2Vec(words, window=5, vector_size=200)
    model.train(words, total_examples=len(words), epochs=200)
    pickle.dump(model, open(word2vec_model_path, 'wb'))
    return model

sentences = get_all_book_sentences(books_dir)
# print(sentences)
sentences = [tokenize_nltk(s.lower()) for s in sentences]

model = train_word2vec(sentences, word2vec_model_path)

w1 = "river"
words = model.wv.most_similar(w1, topn=10)
print(words)