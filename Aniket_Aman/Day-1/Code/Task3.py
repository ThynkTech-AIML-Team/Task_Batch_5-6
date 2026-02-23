import nltk
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

text_doc = """Welcome to India! 🇮🇳 Explore Mumbai, the 'City of Dreams', where the Bollywood industry thrives. 
Visit Pune at info@pune-tourism.in for its rich educational heritage. #EducationHub 🎓
Bengaluru is the Silicon Valley of India; it is famous for its IT sector and traffic!!! 
In Delhi, the capital, you can see historical monuments like the Red Fort. Contact: delhi_help123@gov.in. 
Hyderabad is known for its Biryani and the Charminar. #Foodie #Heritage"""

# --- A. Tokenization ---
print("\n--- Task 3A: Tokenization ---")
# Using NLTK only since Spacy is incompatible with Python 3.14
nltk_tokens = nltk.word_tokenize(text_doc)
print(f"NLTK Tokens: {nltk_tokens}")
print("(Note: Spacy tokenization skipped due to Python 3.14 environment)")

# --- B. Stopwords Removal ---
print("\n--- Task 3B: Stopwords Removal ---")
stop_words = set(stopwords.words('english'))
filtered_text = [w for w in nltk_tokens if w.lower() not in stop_words]
print(f"Original: {nltk_tokens}")
print(f"Filtered: {filtered_text}")

# --- C. Lemmatization & Stemming ---
print("\n--- Task 3C: Lemmatization vs Stemming ---")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = ["studies", "studying", "flies", "running"]

print(f"{'Word':<10} | {'Stem':<10} | {'Lemma'}")
print("-" * 35)
for w in words:
    # 'v' stands for verb, helping lemmatizer understand context
    print(f"{w:<10} | {stemmer.stem(w):<10} | {lemmatizer.lemmatize(w, pos='v')}")

# --- D, E, F. Text Cleaning & Normalization ---
print("\n--- Task 3 D-F: Advanced Cleaning ---")
messy_text = "Contact us at support@example.com!!! I LOVE coding...  #Python 🐍 123"

def clean_text(text):
    # D. Remove Emojis
    text = emoji.replace_emoji(text, replace='')
    
    # F. Extract Emails
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    print(f"Extracted Emails: {emails}")
    
    # E. Lowercasing
    text = text.lower()
    
    # F. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # D. Remove special characters (keep basic punctuation if needed, or remove all)
    text = re.sub(r'[^\w\s]', '', text) 
    
    # F. Replace multiple spaces with one
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

cleaned = clean_text(messy_text)
print(f"Cleaned Text: '{cleaned}'")