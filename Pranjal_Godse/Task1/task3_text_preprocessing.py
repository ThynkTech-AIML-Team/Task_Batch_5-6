import nltk
import spacy
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


nlp = spacy.load("en_core_web_sm")


text = """
Hello @john!!! NLP is AMAZING 😊😊.
Email me at example@gmail.com.
I have studied 3 courses in 2024.
#AI #MachineLearning
"""

print("ORIGINAL TEXT:\n", text)


# A. TOKENIZATION

print("\n--- TOKENIZATION ---")

nltk_tokens = nltk.word_tokenize(text)
print("\nNLTK Tokens:\n", nltk_tokens)

doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("\nspaCy Tokens:\n", spacy_tokens)

# B. STOPWORDS REMOVAL

print("\n--- STOPWORDS REMOVAL ---")

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in nltk_tokens if word.lower() not in stop_words]

print("\nBefore:\n", nltk_tokens)
print("\nAfter:\n", filtered_words)


# C. STEMMING & LEMMATIZATION

print("\n--- STEMMING & LEMMATIZATION ---")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

sample_words = ["studies", "studying", "studied"]

for word in sample_words:
    print(f"\nWord: {word}")
    print("Stem:", stemmer.stem(word))
    print("Lemma:", lemmatizer.lemmatize(word))

# D. CLEANING (Hashtags, Mentions, Emojis, Punctuation)

print("\n--- TEXT CLEANING ---")

clean_text = emoji.replace_emoji(text, replace='')

clean_text = re.sub(r'[@#]\w+', '', clean_text)

clean_text = re.sub(r'[^\w\s]', '', clean_text)

print("\nCleaned Text:\n", clean_text)


# E. LOWERCASE & NORMALIZATION

print("\n--- LOWERCASE & NORMALIZATION ---")

normalized_text = clean_text.lower()
normalized_text = " ".join(normalized_text.split())

print("\nNormalized Text:\n", normalized_text)


# F. REGEX TASKS

print("\n--- REGEX OPERATIONS ---")

emails = re.findall(r'\S+@\S+', text)
print("\nExtracted Emails:\n", emails)

no_numbers = re.sub(r'\d+', '', text)
print("\nText Without Numbers:\n", no_numbers)

single_space = re.sub(r'\s+', ' ', text)
print("\nSingle Spaced Text:\n", single_space)
