import nltk
import re
import emoji
import sys
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- FIX: Force UTF-8 Encoding for Windows Terminal ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # Added for newer NLTK versions

text_doc = "The quick brown fox jumps over the lazy dog. He is studying NLP! #AI @ThynkTech 🚀"

# --- A. Tokenization ---
print("\n--- Task 3A: Tokenization ---")
nltk_tokens = nltk.word_tokenize(text_doc)
print(f"NLTK Tokens: {nltk_tokens}")

# --- B. Stopwords Removal ---
print("\n--- Task 3B: Stopwords Removal ---")
stop_words = set(stopwords.words('english'))
filtered_text = [w for w in nltk_tokens if w.lower() not in stop_words]
print(f"Filtered: {filtered_text}")

# --- C. Lemmatization & Stemming ---
print("\n--- Task 3C: Lemmatization vs Stemming ---")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = ["studies", "studying", "flies", "running"]

for w in words:
    print(f"{w:<10} | {stemmer.stem(w):<10} | {lemmatizer.lemmatize(w, pos='v')}")

# --- D, E, F. Text Cleaning ---
print("\n--- Task 3 D-F: Advanced Cleaning ---")
messy_text = "Contact us at support@example.com!!! I LOVE coding...  #Python 🐍 123"

def clean_text(text):
    text = emoji.replace_emoji(text, replace='') # Remove Emojis
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    print(f"Extracted Emails: {emails}")
    text = text.lower() # Lowercase
    text = re.sub(r'\d+', '', text) # Remove Numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove Special Chars
    text = re.sub(r'\s+', ' ', text).strip() # Clean Spaces
    return text

cleaned = clean_text(messy_text)
print(f"Cleaned Text Result: '{cleaned}'")