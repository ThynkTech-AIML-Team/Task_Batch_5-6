import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "paragraph.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "text_basics_results.txt")

def load_text():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return file.read()

def get_paragraphs(text):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

def get_sentences(text):
    return sent_tokenize(text)

def get_tokens(text):
    return word_tokenize(text)

def save_results(paragraphs, sentences, tokens):

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        file.write("TEXT DATA BASICS RESULTS\n")
        file.write("=========================\n\n")

        file.write("PARAGRAPHS:\n")
        for i, p in enumerate(paragraphs, 1):
            file.write(f"{i}. {p}\n")

        file.write("\nSENTENCES:\n")
        for i, s in enumerate(sentences, 1):
            file.write(f"{i}. {s}\n")

        file.write("\nTOKENS:\n")
        file.write(str(tokens))

        file.write("\n\nCOUNTS:\n")
        file.write(f"Paragraph count: {len(paragraphs)}\n")
        file.write(f"Sentence count: {len(sentences)}\n")
        file.write(f"Token count: {len(tokens)}\n")

def main():

    print("Reading text...")
    text = load_text()

    print("Extracting paragraphs...")
    paragraphs = get_paragraphs(text)

    print("Extracting sentences...")
    sentences = get_sentences(text)

    print("Extracting tokens...")
    tokens = get_tokens(text)

    print("Saving output...")
    save_results(paragraphs, sentences, tokens)

    print("Task completed. Check outputs/text_basics_results.txt")


if __name__ == "__main__":
    main()
