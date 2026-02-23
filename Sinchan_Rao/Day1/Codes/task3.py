import nltk
import spacy
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

text = """Hello @john!!! NLP is AMAZING 😍.
Studies show that machines are running faster.
Contact us at test@gmail.com or support123@yahoo.com.
#AI #MachineLearning 12345"""

print("========== ORIGINAL TEXT ==========\n")
print(text)


print("\n========== A. TOKENIZATION ==========\n")

# NLTK Tokenization
nltk_tokens = word_tokenize(text)
print("NLTK Tokens:")
print(nltk_tokens)

# spaCy Tokenization
doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("\nspaCy Tokens:")
print(spacy_tokens)

print("\n========== B. STOPWORDS REMOVAL ==========\n")
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in nltk_tokens 
                   if word.lower() not in stop_words]

print("Before Removal:")
print(nltk_tokens)

print("\nAfter Removal:")
print(filtered_tokens)

print("\n========== C. STEMMING vs LEMMATIZATION ==========\n")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = ["studies", "running", "better"]
for word in words:
    print("Word:", word)
    print("Stem:", stemmer.stem(word))
    print("Lemma:", lemmatizer.lemmatize(word))
    print()

print("\n========== D. CLEANING SPECIAL CHARACTERS ==========\n")
# Remove emojis
clean_text = emoji.replace_emoji(text, replace='')
# Remove hashtags and mentions
clean_text = re.sub(r'[@#]\w+', '', clean_text)
# Remove punctuation
clean_text = re.sub(r'[^\w\s]', '', clean_text)
print("Cleaned Text:")
print(clean_text)

print("\n========== E. LOWERCASE & NORMALIZATION ==========\n")
normalized_text = clean_text.lower()
normalized_text = " ".join(normalized_text.split())
print("Normalized Text:")
print(normalized_text)

print("\n========== F. REGEX OPERATIONS ==========\n")
# Extract emails
emails = re.findall(r'\S+@\S+', text)
print("Extracted Emails:", emails)
# Remove numbers
no_numbers = re.sub(r'\d+', '', text)
print("\nText Without Numbers:")
print(no_numbers)
# Replace multiple spaces
single_space = re.sub(r'\s+', ' ', text)
print("\nText With Normalized Spaces:")
print(single_space)
print("\n========== TASK COMPLETED ==========")
