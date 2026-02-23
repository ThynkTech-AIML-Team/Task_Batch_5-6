import re
import nltk
import spacy
import emoji

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

print(">>> Script started")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Hello @Krish!!! NLP is AMAZING 😍😍 #AI #MachineLearning 123 studies studying studies."

print("\n================ RAW TEXT ================\n")
print(text)


# ------------------------------------------
# A. Tokenization (NLTK vs spaCy)
# ------------------------------------------
print("\n========== TOKENIZATION ==========\n")

nltk_tokens = nltk.word_tokenize(text)
print("NLTK Tokens:")
print(nltk_tokens)

doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("\nspaCy Tokens:")
print(spacy_tokens)


# ------------------------------------------
# B. Stopwords Removal
# ------------------------------------------
print("\n========== STOPWORDS REMOVAL ==========\n")

stop_words = set(stopwords.words("english"))
filtered_words = [w for w in nltk_tokens if w.lower() not in stop_words]

print("Before:", nltk_tokens)
print("After:", filtered_words)


# ------------------------------------------
# C. Lemmatization & Stemming
# ------------------------------------------
print("\n========== STEMMING vs LEMMATIZATION ==========\n")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "studies"

print("Original:", word)
print("Stemmed:", stemmer.stem(word))
print("Lemmatized:", lemmatizer.lemmatize(word))


# ------------------------------------------
# D. Cleaning hashtags, mentions, emojis
# ------------------------------------------
print("\n========== CLEANING SPECIAL TEXT ==========\n")

clean_text = re.sub(r"@\w+|#\w+", "", text)
clean_text = emoji.replace_emoji(clean_text, replace="")
clean_text = re.sub(r"[!]+", "!", clean_text)

print("Cleaned text:")
print(clean_text)


# ------------------------------------------
# E. Lowercasing & Normalization
# ------------------------------------------
print("\n========== LOWERCASE & NORMALIZATION ==========\n")

normalized_text = " ".join(clean_text.lower().split())
print(normalized_text)


# ------------------------------------------
# F. Regex Operations
# ------------------------------------------
print("\n========== REGEX TASKS ==========\n")

paragraph = "Contact us at krish@gmail.com or hr@company.in. Call 9876543210."

emails = re.findall(r'\S+@\S+', paragraph)
print("Emails found:", emails)

no_numbers = re.sub(r'\d+', '', paragraph)
print("\nText without numbers:")
print(no_numbers)

single_space = re.sub(r'\s+', ' ', paragraph)
print("\nNormalized spacing:")
print(single_space)

print("\n>>> Script finished")
