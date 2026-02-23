import nltk

# -----------------------------
# Text Analysis Function
# -----------------------------
def analyze_text(paragraph):

    # Sentence tokenization
    sentences = nltk.sent_tokenize(paragraph)

    # Word tokenization
    tokens = nltk.word_tokenize(paragraph)

    # Paragraph split (based on new lines)
    paragraphs = [p for p in paragraph.split("\n") if p.strip() != ""]

    print("\n--- Text Analysis ---")

    print("\nSentences:")
    print(sentences)

    print("\nTokens:")
    print(tokens)

    print("\nCounts:")
    print("Sentence count:", len(sentences))
    print("Token count:", len(tokens))
    print("Paragraph count:", len(paragraphs))


# -----------------------------
# Example Text
# -----------------------------
text = """NLP is amazing. It allows computers to understand human language.
This is your internship task."""

analyze_text(text)
