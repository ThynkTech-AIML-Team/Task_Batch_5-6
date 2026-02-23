import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("\n=== Task 2: Analyzing Text ===")
print()

def analyze_text(paragraph):
    # split into sentences
    sentences = sent_tokenize(paragraph)
    
    # split into tokens/words
    tokens = word_tokenize(paragraph)
    
    # count everything
    num_sentences = len(sentences)
    num_tokens = len(tokens)
    num_paragraphs = len([p for p in paragraph.split('\n') if p.strip()])
    
    return {
        'sentences': sentences,
        'tokens': tokens,
        'num_sentences': num_sentences,
        'num_tokens': num_tokens,
        'num_paragraphs': num_paragraphs
    }

# testing with example text
example_paragraph = """Natural Language Processing (NLP) is a fascinating field. It enables computers to understand human language!
NLP is used in chatbots, translation, and more."""

print("example text:")
print(example_paragraph)
print()

result = analyze_text(example_paragraph)

print("\nwhat we got:\n")

print("sentences found:")
for i, sentence in enumerate(result['sentences'], 1):
    print(f"  {i}. {sentence}")

print(f"\ntokens: {result['tokens']}")

print(f"\ncounts:")
print(f"  sentences: {result['num_sentences']}")
print(f"  tokens: {result['num_tokens']}")
print(f"  paragraphs: {result['num_paragraphs']}")
