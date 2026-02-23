# Task 2: Text Data Basics

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def text_data_basics(paragraph):
    sentences = sent_tokenize(paragraph)
    
    tokens = word_tokenize(paragraph)
    
    sentence_count = len(sentences)
    token_count = len(tokens)
    paragraph_count = 1

    return {
        "Sentences": sentences,
        "Tokens": tokens,
        "Sentence Count": sentence_count,
        "Token Count": token_count,
        "Paragraph Count": paragraph_count
    }

text = "Natural Language Processing is interesting. It helps computers understand human language."

result = text_data_basics(text)


print("Sentences:")
print(result["Sentences"])

print("\nTokens:")
print(result["Tokens"])

print("\nCounts:")
print("Sentences:", result["Sentence Count"])
print("Tokens:", result["Token Count"])
print("Paragraphs:", result["Paragraph Count"])