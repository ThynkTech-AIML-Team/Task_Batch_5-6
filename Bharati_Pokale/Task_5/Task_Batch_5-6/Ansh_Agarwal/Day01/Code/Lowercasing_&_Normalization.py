import re

text = "  NLP   Is   VERY    Powerful   "

text = text.lower()

text = re.sub(r'\s+', ' ', text).strip()

print(text)