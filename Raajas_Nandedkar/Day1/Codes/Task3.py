# =========================================
# TASK 3 — TEXT PREPROCESSING
# A — Tokenization
# B — Stopwords Removal
# C — Stemming vs Lemmatization
# D — Cleaning special chars/emojis
# E — Lowercase & normalization
# F — Regex tasks
# =========================================
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import emoji


# -----------------------------
# PART A — TOKENIZATION
# -----------------------------
print("\n===== TASK 3A — TOKENIZATION =====")

text = "I can't believe NLP-based systems work!"

print("word_tokenize:", word_tokenize(text))
print("wordpunct_tokenize:", wordpunct_tokenize(text))


# -----------------------------
# PART B — STOPWORDS REMOVAL
# -----------------------------
print("\n===== TASK 3B — STOPWORDS REMOVAL =====")

stop_words = set(stopwords.words("english"))
tokens = word_tokenize("This is a simple example showing stopword removal")

filtered = [w for w in tokens if w.lower() not in stop_words]

print("Before:", tokens)
print("After:", filtered)


# -----------------------------
# PART C — STEMMING vs LEMMA
# -----------------------------
print("\n===== TASK 3C — STEMMING vs LEMMATIZATION =====")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = word_tokenize("studies studying studied study")

for w in words:
    print(w,
          "Stem:", stemmer.stem(w),
          "Lemma:", lemmatizer.lemmatize(w))


# -----------------------------
# PART D — CLEAN TEXT
# -----------------------------
print("\n===== TASK 3D — CLEAN SPECIALS =====")

dirty = "Hello!!! 😊 Check #NLP @user now!!!"

clean = emoji.replace_emoji(dirty, "")
clean = re.sub(r'[@#]\w+', '', clean)
clean = re.sub(r'[^\w\s]', '', clean)

print("Cleaned:", clean)


# -----------------------------
# PART E — LOWERCASE + NORMALIZE
# -----------------------------
print("\n===== TASK 3E — NORMALIZATION =====")

mixed = "   NLP   Is   VERY   COOL   "
normalized = " ".join(mixed.lower().split())

print("Normalized:", normalized)


# -----------------------------
# PART F — REGEX TASKS
# -----------------------------
print("\n===== TASK 3F — REGEX =====")

sample = """
Contact: help@test.com and admin@mail.org
Numbers 12345 present
Extra    spaces here
"""

emails = re.findall(r'\S+@\S+', sample)
print("Emails:", emails)

print("No numbers:", re.sub(r'\d+', '', sample))
print("Single spaces:", re.sub(r'\s+', ' ', sample))