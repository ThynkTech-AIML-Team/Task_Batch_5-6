
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


# Task A -----------------------------------------------------------------------------
# install spacy and run "python -m spacy download en_core_web_sm"


import spacy

nltk.download('punkt')
print("\nTask A — Tokenization: ------------------------------------------")
text = "Dr. Smith's well-known e-mail address is dr.smith@example.com."
print("Sample Text: "+text)


# NLTK tokenizer
nltk_tokens = word_tokenize(text)
print("NLTK Tokens:", nltk_tokens)

# spaCy tokenizer
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("spaCy Tokens:", spacy_tokens)

# Task B ---------------------------------------------------------------------
print("\nTask B — Stopwords Removal: ------------------------------------------")

stop_words = set(stopwords.words("english"))
tokens = word_tokenize("This is a simple example showing stopword removal")
print("Sample Text: This is a simple example showing stopword removal ")

filtered = [w for w in tokens if w.lower() not in stop_words]

print("Before:", tokens)
print("After:", filtered)

#Task C
print("\nTask C — Stemming vs Lemmatization: ------------------------------------------")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = word_tokenize("Studies of children’s behaviors show that they often run and play outside.")
print("Sample Text: Studies of children’s behaviors show that they often run and play outside.")
for w in words:
    print(w+"-->",
          "Stem:", stemmer.stem(w),"| "
          "Lemma:", lemmatizer.lemmatize(w))

#Task D
print("\nTask D — Cleaning special chars/emojis: ------------------------------------------")
sample = "Hello @user! I love ThynkTech!!! 😍 #coding #AIML 🤖 Let's see how this works.!!!"
print("Sample Text: " + sample)

#Remove @mentions
clean = re.sub(r'@\w+', '', sample)

#Remove #hashtags
clean = re.sub(r'#\w+', '', clean)

#Remove multiple exclamation marks (or excessive punctuation)
clean = re.sub(r'!+', '!', clean)  # replace multiple ! with single !
clean = re.sub(r'[^\w\s!?.]', '', clean)  # remove other special characters except basic punctuation

#Remove emojis
clean = emoji.replace_emoji(clean, replace='')

#Remove extra whitespace
clean = re.sub(r'\s+', ' ', clean).strip()

print(clean)

#Task E
print("\nTask E — Lowercase & normalization: ------------------------------------------")

sample_text2 = "   This   is  MIXED-case   Text.  It  needs   Normalization!   "
print("Sample Text: "+ sample_text2)

normalised_text = sample_text2.lower()
normalised_text = re.sub(r'\s+', ' ', normalised_text).strip()

print(normalised_text)

#Task F
print("\nTask  F — Regex tasks: ------------------------------------------")

sample_text3 = """
Contact: thynktech@test.com and admin@mail.org
Phone Number: 9090909090
Extra        spaces here
"""
print("Sample Text : "+ sample_text3)

emails = re.findall(r'\S+@\S+', sample_text3)
print("\nExtracted Emails:", emails)

print("\nText with no numbers:", re.sub(r'\d+', '', sample_text3))
print("\nText with no multiple spaces:", re.sub(r'\s+', ' ', sample_text3))