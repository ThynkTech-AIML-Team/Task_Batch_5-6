# 3(A) Tokenization

import nltk
import spacy
from nltk.tokenize import word_tokenize

nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

text = "Natural Language Processing is amazing, isn't it?"

nltk_tokens = word_tokenize(text)
spacy_tokens = [token.text for token in nlp(text)]

print("=== 3(A) TOKENIZATION ===")
print("Original Text:", text)
print("NLTK Tokens:", nltk_tokens)
print("spaCy Tokens:", spacy_tokens)
