from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import pickle

word2vec_model_path = "./word2vec.model"
model = pickle.load(open(word2vec_model_path, 'rb'))
(analogy_score, word_list) = model.wv.evaluate_word_analogies(datapath('/home/kuan/Desktop/NLP/Chapter03/questions-words.txt'))

print(analogy_score)

pretrained_model_path = "./Chapter03/40/model.bin"
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)
(analogy_score, word_list) = pretrained_model.evaluate_word_analogies(datapath('/home/kuan/Desktop/NLP/Chapter03/questions-words.txt'))

print(analogy_score)

