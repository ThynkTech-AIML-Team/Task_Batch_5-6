import nltk
import re
import string
import emoji
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# load spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))

print("\n=== Task 3: Text Preprocessing ===")
print()

# A. Tokenization
print("A. Tokenization")

sample_text = "NLP includes tokenization, stemming, and lemmatization! Let's test: studies, studying, studied. #NLP @user 😊!!!"

print(f"text: {sample_text}\n")

# nltk way
nltk_tokens = word_tokenize(sample_text)
print("nltk tokens:")
print(f"  {nltk_tokens}\n")

# spacy way
spacy_doc = nlp(sample_text)
spacy_tokens = [token.text for token in spacy_doc]
print("spacy tokens:")
print(f"  {spacy_tokens}\n")

print(f"same? {nltk_tokens == spacy_tokens}")
print(f"nltk got {len(nltk_tokens)}, spacy got {len(spacy_tokens)}\n")

# B. Stopwords Removal
print("B. Stopwords Removal")

words = [w for w in nltk_tokens if w.isalpha()]
print("before:")
print(f"  {words}")
print(f"  {len(words)} words\n")

filtered = [w for w in words if w.lower() not in stop_words]
print("after:")
print(f"  {filtered}")
print(f"  {len(filtered)} words\n")

# C. Lemmatization & Stemming
print("C. Lemmatization & Stemming")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words_to_test = ["studies", "studying", "studied"]
for w in words_to_test:
    stem = stemmer.stem(w)
    lemma = lemmatizer.lemmatize(w, pos='v')
    print(f"  {w} -> stem: {stem}, lemma: {lemma}")
print("  stemming is rough, lemmatization is better\n")

# D. Cleaning stuff
print("D. Cleaning special chars & emojis")

text_dirty = "Check this out!!! #awesome @friend 😊"
print(f"before: {text_dirty}")

cleaned = re.sub(r"[@#]\w+", "", text_dirty)  # remove hashtags/mentions
cleaned = re.sub(rf"[{re.escape(string.punctuation)}]", "", cleaned)
cleaned = emoji.replace_emoji(cleaned, replace='')

print(f"after: {cleaned.strip()}\n")

# E. Lowercase and fix spacing
print("E. Lowercasing & Normalization")

mixed_case = "  This   is   a   TEST   "
print(f"before: {mixed_case}")

lowered = mixed_case.lower()
normalized = re.sub(r'\s+', ' ', lowered).strip()
print(f"after: {normalized}\n")

# F. Using regex
print("F. Regex for text cleaning")

# finding emails
text_with_email = "Contact us at info@example.com or support@domain.org."
print(f"text: {text_with_email}")
emails = re.findall(r"[\w\.-]+@[\w\.-]+", text_with_email)
print(f"emails found: {emails}\n")

# removing numbers
text_with_numbers = "There are 123 apples and 45 oranges."
print(f"before: {text_with_numbers}")
no_numbers = re.sub(r"\d+", "", text_with_numbers)
print(f"after: {no_numbers}\n")

# fixing extra spaces
messy = "This    sentence   has   extra   spaces."
print(f"before: {messy}")
clean_spaces = re.sub(r"\s+", " ", messy)
print(f"after: {clean_spaces}")
