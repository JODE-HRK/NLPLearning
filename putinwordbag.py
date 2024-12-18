from sklearn.feature_extraction.text import CountVectorizer
from Chapter01.dividing_into_sentences import read_text_file, preprocess_text, divide_into_sentences_nltk

def get_sentences(filename):
    text = read_text_file(filename)
    text = preprocess_text(text)
    sentences = divide_into_sentences_nltk(text)
    return sentences

def create_vectorizer(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return (vectorizer, X)

sentences = get_sentences("sherlock_holmes_1.txt")
(vectorizer, X) = create_vectorizer(sentences)
print(X)

denseX = X.todense()
print(denseX)

print(vectorizer.get_feature_names())


print("/////////////////////////////////////////////")
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit(sentences)

print(vectorizer.get_feature_names())

new_sentence = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
new_sentence_vector = vectorizer.transform([new_sentence])

analyze = vectorizer.build_analyzer()
print(analyze(new_sentence))
print(new_sentence_vector)


print("/////////////////////////////////////////////")
vectorizer = CountVectorizer(max_df=0.8)
X = vectorizer.fit(sentences)

print(vectorizer.get_feature_names())