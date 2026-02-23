import nltk
import spacy
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "This world is beautiful! Lets explore it together."

# NLTK Tokenization
nltk_tokens = word_tokenize(text)

# spaCy Tokenization 
doc = nlp(text)
spacy_tokens = [token.text for token in doc]

# Print results
print("Original Text:\n", text)

print("\nNLTK Tokens:")
print(nltk_tokens)

print("\nspaCy Tokens:")
print(spacy_tokens)