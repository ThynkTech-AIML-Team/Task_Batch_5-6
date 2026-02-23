from dataset_loader import load_data


import spacy
import re
from dataset_loader import load_data

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    # lowercase
    text = text.lower()

    # remove numbers & special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    return text


def preprocess_documents(docs):

    processed_docs = []

    for doc in docs:
        doc = clean_text(doc)
        spacy_doc = nlp(doc)

        tokens = [
            token.lemma_
            for token in spacy_doc
            if token.is_stop == False
            and len(token) > 2
        ]

        processed_docs.append(" ".join(tokens))

    return processed_docs


if __name__ == "__main__":
    print("Loading dataset...")
    documents = load_data()

    print("Preprocessing started...")
    processed = preprocess_documents(documents[:50])  # small batch for testing

    print("\nSample processed text:\n")
    print(processed[0][:300])
