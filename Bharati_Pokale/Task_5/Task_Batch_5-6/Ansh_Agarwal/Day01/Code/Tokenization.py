import nltk
import spacy
from nltk.tokenize import word_tokenize

text = "NLP is amazing! It helps computers understand human language."

nltk_tokens = word_tokenize(text)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_tokens = [token.text for token in doc]

print("NLTK Tokens:")
print(nltk_tokens)

print("\nspaCy Tokens:")
print(spacy_tokens)