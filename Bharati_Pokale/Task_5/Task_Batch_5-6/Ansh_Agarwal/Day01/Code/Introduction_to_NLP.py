import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]

train_set = featuresets[:1500]
test_set = featuresets[1500:]

classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", accuracy(classifier, test_set))

classifier.show_most_informative_features(5)