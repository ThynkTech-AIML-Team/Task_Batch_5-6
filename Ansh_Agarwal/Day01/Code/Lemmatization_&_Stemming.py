from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

text = "studies studying studied"
tokens = word_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Word | Stem | Lemma")
for word in tokens:
    print(word, "|", stemmer.stem(word), "|", lemmatizer.lemmatize(word))