import nltk
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

#Dataset Download
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('punkt_tab')



positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = positive_tweets + negative_tweets
labels = ['positive'] * len(positive_tweets) + ['negative'] * len(negative_tweets)

#Model Training
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(tweets)

model = MultinomialNB()
model.fit(X, labels)


#User Input
user_text = input("Enter a tweet or paragraph:\n")


#Sentiment Analysis Output
user_vector = vectorizer.transform([user_text])
prediction = model.predict(user_vector)[0]

print("\nSentiment Analysis Output :")
print("Sentiment:", prediction.capitalize())


#Text Summarization Output
print("\nText Summarization Output :")

parser = PlaintextParser.from_string(user_text, Tokenizer("english"))
summarizer = LsaSummarizer()

summary = summarizer(parser.document, 2)  #summary length fixed at 2 sentences.

for sentence in summary:
    print("-", sentence)
