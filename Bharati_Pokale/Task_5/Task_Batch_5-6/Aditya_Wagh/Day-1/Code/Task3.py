import sys
sys.stdout.reconfigure(encoding='utf-8')

import nltk
import spacy
import re
import emoji


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

text = "Hello @user!!! This is #NLP 😊. Studies are running 123. Email me at test@gmail.com"

print("Original Text:")
print(text)

print("\n--- Tokenization ---")

nltk_tokens = word_tokenize(text)
print("NLTK Tokens:", nltk_tokens)

doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("spaCy Tokens:", spacy_tokens)

print("\n--- Stopwords Removal ---")

stop_words = set(stopwords.words("english"))

filtered_tokens = [word for word in nltk_tokens if word.lower() not in stop_words]

print("Before:", nltk_tokens)
print("After:", filtered_tokens)


print("\n--- Stemming and Lemmatization ---")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["studies", "running", "playing"]

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    print(word, "-> Stemmed:", stem, "| Lemmatized:", lemma)

print("\n--- Cleaning Text ---")

clean_text = re.sub(r'[@#]\w+', '', text)  # remove mentions and hashtags
clean_text = re.sub(r'[^\w\s@.]', '', clean_text)  # remove punctuation
clean_text = emoji.replace_emoji(clean_text, replace='')  # remove emoji

print("Cleaned Text:", clean_text)

print("\n--- Lowercase and Normalization ---")

lower_text = clean_text.lower()

normalized_text = re.sub(r'\s+', ' ', lower_text)

print("Normalized Text:", normalized_text)

print("\n--- Regex Operations ---")

emails = re.findall(r'\S+@\S+', text)
print("Emails Found:", emails)

no_numbers = re.sub(r'\d+', '', text)
print("Text without numbers:", no_numbers)

clean_spaces = re.sub(r'\s+', ' ', text)
print("Text with single spaces:", clean_spaces)
