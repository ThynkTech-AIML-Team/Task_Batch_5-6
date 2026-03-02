import re

def quick_text_analysis(text: str) -> dict:
    """
    Simple text analysis that takes a paragraph (or longer text)
    and returns sentences, tokens, and counts.
    
    Args:
        text: input paragraph or multi-paragraph text
        
    Returns:
        dict with keys: 'sentences', 'tokens', 'counts'
    """
    if not text or not text.strip():
        return {
            "sentences": [],
            "tokens": [],
            "counts": {"tokens": 0, "sentences": 0, "paragraphs": 0}
        }
    
    #Paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    #Sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 1]
    
    #Tokens
    tokens = re.findall(r"\b\w+(?:['’-]\w+)?\b", text)
    
    return {
        "sentences": sentences,
        "tokens": tokens,
        "counts": {
            "tokens": len(tokens),
            "sentences": len(sentences),
            "paragraphs": max(1, len(paragraphs))
        }
    }

paragraph = """
Vincent van Gogh (1853–1890) remains one of the most influential
and recognizable figures in art history, a Dutch Post-Impressionist painter whose intense emotional
expression, bold use of color, and dynamic brushwork helped pave the way for modern art
movements like Expressionism.

He painted what he felt rather than what he saw exactly
"""

result = quick_text_analysis(paragraph)

print("Sentences:", result["sentences"])
print("Tokens count:", result["counts"]["tokens"])
print("Sentences count:", result["counts"]["sentences"])
print("Paragraphs count:", result["counts"]["paragraphs"])