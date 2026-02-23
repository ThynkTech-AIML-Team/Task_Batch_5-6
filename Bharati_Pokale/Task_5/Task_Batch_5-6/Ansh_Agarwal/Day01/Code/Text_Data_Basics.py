import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def analyze_text(paragraph_text):
    paragraphs = [p for p in paragraph_text.split('\n') if p.strip() != ""]
    
    sentences = sent_tokenize(paragraph_text)
    
    tokens = word_tokenize(paragraph_text)
    
    sentence_count = len(sentences)
    token_count = len(tokens)
    paragraph_count = len(paragraphs)
    
    return {
        "Sentences": sentences,
        "Tokens": tokens,
        "Sentence Count": sentence_count,
        "Token Count": token_count,
        "Paragraph Count": paragraph_count
    }



text = """Natural Language Processing is very interesting. 
It allows computers to understand human language.

It is widely used in chatbots and sentiment analysis."""

result = analyze_text(text)

for key, value in result.items():
    print(f"\n{key}:")
    print(value)