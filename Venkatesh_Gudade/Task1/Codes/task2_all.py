# ==============================
# TASK 2 — TEXT DATA BASICS
# ==============================

from nltk.tokenize import sent_tokenize, word_tokenize

print("\n===== TASK 2 — TEXT ANALYSIS =====")

paragraph = """
Natural Language Processing is interesting. It allows machines to read text.
It is widely used in many applications.
"""

# Sentences
sentences = sent_tokenize(paragraph)

# Tokens
tokens = word_tokenize(paragraph)

print("\nSentences:")
for s in sentences:
    print("-", s)

print("\nTokens:")
print(tokens)

print("\nCounts:")
print("Sentence count:", len(sentences))
print("Token count:", len(tokens))
print("Paragraph count:", len(paragraph.split("\n\n")))
