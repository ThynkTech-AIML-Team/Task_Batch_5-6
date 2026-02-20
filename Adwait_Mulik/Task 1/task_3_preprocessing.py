import re
import nltk
import spacy
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load necessary NLP models and resources
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def run_task_3():
    print("--- STARTING TASK 3: TEXT PREPROCESSING ---")
    
    # Sample noisy text for demonstration
    raw_text = """Hello #TeamThynkTech! Check our new AI modules at info@thynktech.in. 
    The students are studying 3 different NLP courses!!! 🚀 @adwait, stay tuned for more.   """

    # --- A. TOKENIZATION COMPARISON ---
    # [cite: 15, 18] Compare NLTK word_tokenizer vs spaCy tokenizer
    print("\n[A] Tokenization Comparison:")
    nltk_tokens = nltk.word_tokenize(raw_text)
    spacy_tokens = [token.text for token in nlp(raw_text)]
    print(f"NLTK Tokens (First 10): {nltk_tokens[:10]}")
    print(f"spaCy Tokens (First 10): {spacy_tokens[:10]}")

    # --- B. STOPWORDS REMOVAL ---
    # [cite: 19, 20] Remove common words and print before vs after
    print("\n[B] Stopwords Removal:")
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in nltk_tokens if w.lower() not in stop_words]
    print(f"Before: {len(nltk_tokens)} tokens")
    print(f"After: {len(filtered_tokens)} tokens")

    # --- C. LEMMATIZATION & STEMMING ---
    # [cite: 21, 23] Apply PorterStemmer and WordNetLemmatizer
    print("\n[C] Stemming vs Lemmatization:")
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    word = "studying" # Requirements example: "studies" -> "studi" vs "study" 
    print(f"Original: {word}")
    print(f"Stemmed: {ps.stem(word)}")
    print(f"Lemmatized: {wnl.lemmatize(word, pos='v')}")

    # --- D, E, & F. CLEANING, NORMALIZATION & REGEX ---
    # [cite: 24, 28, 29, 30] Handling punctuation, emojis, and mixed-case
    print("\n[D, E, F] Regex Cleaning & Normalization:")
    
    # 1. Extract Emails [cite: 32]
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', raw_text)
    print(f"Extracted Emails: {emails}")

    # 2. Lowercasing & Emoji Normalization [cite: 28, 29]
    clean_text = raw_text.lower()
    clean_text = emoji.demojize(clean_text) # Converts 🚀 to :rocket:

    # 3. Removing numbers and cleaning hashtags/mentions [cite: 26, 33]
    clean_text = re.sub(r'#\S+|@\S+', '', clean_text) # Remove #hashtags and @mentions
    clean_text = re.sub(r'\d+', '', clean_text) # Remove numbers
    clean_text = re.sub(r'[^\w\s]', '', clean_text) # Remove punctuation

    # 4. Normalize Spacing [cite: 34]
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    print(f"Final Cleaned Text: {clean_text}")

    # Save output to satisfy office GitHub requirements
    with open('outputs/task_3_cleaned_report.txt', 'w') as f:
        f.write(f"Emails found: {emails}\n")
        f.write(f"Cleaned Version: {clean_text}")
    print("\n[SUCCESS] Preprocessing report saved in 'outputs/task_3_cleaned_report.txt'.")

if __name__ == "__main__":
    run_task_3()