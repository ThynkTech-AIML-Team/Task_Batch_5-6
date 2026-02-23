import re
import nltk
import spacy
import emoji

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Run once:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# python -m spacy download en_core_web_sm


nlp = spacy.load("en_core_web_sm")


text = "Hello! I'm X. NLP is awesome, isn't it?"

nltk_tokens = word_tokenize(text)
spacy_tokens = [token.text for token in nlp(text)]

print("\n--- A) TOKENIZATION ---")
print("NLTK Tokens:", nltk_tokens)
print("spaCy Tokens:", spacy_tokens)


stop_words = set(stopwords.words("english"))

tokens = word_tokenize(text)
filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]

print("\n--- B) STOPWORDS REMOVAL ---")
print("Before:", tokens)
print("After:", filtered)


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

sample = "Studies studying study studied easily running runs ran."
sample_tokens = word_tokenize(sample)

stems = [stemmer.stem(w) for w in sample_tokens]
lemmas = [lemmatizer.lemmatize(w) for w in sample_tokens]

print("\n--- C) STEMMING vs LEMMATIZATION ---")
print("Original:", sample_tokens)
print("Stemming:", stems)
print("Lemmatization:", lemmas)


def clean_social_text(text):
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

social = "OMG!!! I love #NLP @OpenAI!!! This is sooo cool!!!"

print("\n--- D) CLEAN SOCIAL TEXT ---")
print("Before:", social)
print("After:", clean_social_text(social))


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

mix = "   HeLLo   ShRaVaNi   HOW   Are   You??   "

print("\n--- E) LOWERCASE + NORMALIZE ---")
print("Before:", mix)
print("After:", normalize_text(mix))


regex_text = """
Contact us at support@gmail.com or hr@company.co.in.
My phone number is 9876543210 and I have 2 projects.
"""

emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", regex_text)
no_numbers = re.sub(r"\d+", "", regex_text)
clean_spaces = re.sub(r"\s+", " ", no_numbers).strip()

print("\n--- F) REGEX CLEANING ---")
print("Extracted Emails:", emails)
print("Text without numbers:", clean_spaces)
