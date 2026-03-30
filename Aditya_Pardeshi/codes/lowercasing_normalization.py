import re

def normalize_text(text):
    print("Original Text:\n", text)

    # 1️⃣ Convert to lowercase
    text = text.lower()

    # 2️⃣ Normalize spacing (remove extra spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)

    # 3️⃣ Remove leading & trailing spaces
    text = text.strip()

    print("\nNormalized Text:\n", text)

    return text


# Example text
sample_text = "   AI   Is   AMAZING!!!    NLP   Is   Powerful.   "

normalize_text(sample_text)
