import re

def preprocess_text(documents):
    """
    Lowercase text and remove non-alphanumeric characters.
    """
    processed_docs = []
    for doc in documents:
        text = doc.lower()
        text = re.sub(r'\W+', ' ', text)
        processed_docs.append(text)
    return processed_docs
