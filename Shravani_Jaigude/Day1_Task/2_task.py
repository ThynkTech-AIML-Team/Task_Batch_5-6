import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def text_data_basics(paragraph):
    paragraphs = paragraph.strip().split("\n\n")
    para_count = len(paragraphs)

    sentences = sent_tokenize(paragraph)
    tokens = word_tokenize(paragraph)

    return {
        "paragraph_count": para_count,
        "sentences": sentences,
        "sentence_count": len(sentences),
        "tokens": tokens,
        "token_count": len(tokens),
    }


if __name__ == "__main__":
    text = """NLP is a fascinating field. It helps computers understand human language.
It is used in chatbots, translation, and sentiment analysis."""

    result = text_data_basics(text)

    print("Paragraph Count:", result["paragraph_count"])
    print("Sentence Count:", result["sentence_count"])
    print("\nSentences:")
    for i, s in enumerate(result["sentences"], start=1):
        print(f"{i}. {s}")

    print("\nToken Count:", result["token_count"])
    print("Tokens:", result["tokens"])
