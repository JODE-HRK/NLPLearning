import nltk
from nltk.probability import FreqDist

filename = "sherlock_holmes.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()
text = text.replace("\n", " ")
words = nltk.tokenize.word_tokenize(text)

freq_dist = FreqDist(word.lower() for word in words)
words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]

sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])

print(sorted_words[0])

stopwords = [tuple[0] for tuple in sorted_words if tuple[1] > 100] # if frequency > 100, can be stopword
print(stopwords)

length_cutoff = int(0.02*len(sorted_words)) # sort the word based on the frequency, take the 0.02% most frequent word as the stopword
stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]
print(stopwords)