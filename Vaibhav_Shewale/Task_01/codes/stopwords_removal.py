import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    
    filtered_text = [w for w in word_tokens if w.lower() not in stop_words]
    
    return " ".join(filtered_text)

original_text = "This world is huge, and is inhabited by millions of species."
cleaned_text = clean_text(original_text)

print(f"--- Before ---\n{original_text}")
print(f"\n--- After ---\n{cleaned_text}")