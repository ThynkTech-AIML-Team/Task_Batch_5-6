
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def analyze_text(paragraph):
    # Paragraph count (split by empty lines)
    paragraphs = [p for p in paragraph.split("\n") if p.strip() != ""]
    paragraph_count = len(paragraphs)

    # Sentence tokenization
    sentences = sent_tokenize(paragraph)
    sentence_count = len(sentences)

    # Word tokenization
    tokens = word_tokenize(paragraph)
    
    # Remove punctuation tokens (optional improvement)
    words = [word for word in tokens if word.isalnum()]
    token_count = len(words)

    return {
        "sentences": sentences,
        "tokens": words,
        "sentence_count": sentence_count,
        "token_count": token_count,
        "paragraph_count": paragraph_count
    }


# Example Usage
text = """Artificial Intelligence is transforming industries.
It improves efficiency and accuracy.
AI applications include chatbots, recommendation systems, and autonomous vehicles."""

result = analyze_text(text)

print("Sentences:\n", result["sentences"])
print("\nTokens:\n", result["tokens"])
print("\nSentence Count:", result["sentence_count"])
print("Token Count:", result["token_count"])
print("Paragraph Count:", result["paragraph_count"])