import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def analyze_text(paragraph):

    sentences = sent_tokenize(paragraph)
    tokens = word_tokenize(paragraph)

    print("List of Sentences:")
    print(sentences)

    print("\nList of Tokens:")
    print(tokens)

    print("\nNumber of Sentences:", len(sentences))
    print("Number of Tokens:", len(tokens))
    print("Number of Paragraphs:", len(paragraph.split("\n")))



text = """Natural Language Processing is a field of Artificial Intelligence.
It helps computers understand human language.
It is used in chatbots and translation systems."""

analyze_text(text)
