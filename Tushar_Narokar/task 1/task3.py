
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

def sep(title): print(f"\n{'='*55}\n  {title}\n{'='*55}")

def spacy_like_tokenize(text):
    text = re.sub(r"(\w)'(\w)",  r"\1 '\2", text)   
    text = re.sub(r"(\w)-(\w)",  r"\1 - \2", text)  
    text = re.sub(r"([^\w\s])",  r" \1 ",    text)  
    return [t for t in text.split() if t]




print("A. TOKENIZATION----------------------------")
text = "The quick brown fox jumps over the lazy dog! It's a well-known sentence, often used in NLP tasks."

nltk_tokens  = word_tokenize(text)
spacy_tokens = spacy_like_tokenize(text)

print(f"NLTK  ({len(nltk_tokens)} tokens): {nltk_tokens}")
print(f"spaCy ({len(spacy_tokens)} tokens): {spacy_tokens}")
print("\n")




print("B. STOPWORDS REMOVAL----------------------------")
text   = "This is a simple example of how stopwords are removed from a piece of text."
stop   = set(stopwords.words('english'))
before = word_tokenize(text)
after  = [w for w in before if w.lower() not in stop]

print(f"Before ({len(before)}): {before}")
print(f"After  ({len(after)}):  {after}")
print("\n")



print("C. STEMMING & LEMMATIZATION-----------------------------")
words = ["studies", "studying", "studied", "running", "better",
         "wolves", "leaves", "caring", "beautifully"]
ps, wnl = PorterStemmer(), WordNetLemmatizer()

print(f"{'Word':<14} {'Stem':<14} {'Lemma'}")
print("-" * 40)
for w in words:
    print(f"{w:<14} {ps.stem(w):<14} {wnl.lemmatize(w, pos='v')}")
print("\n")





print("D. PUNCTUATION, SPECIAL CHARS------------------------------")
raw = "OMG!!! @JohnDoe posted on #MachineLearning  Visit https://example.com It's 50% off!! "
print(f"Raw:     {raw}")

cleaned = re.sub(r'https?://\S+|www\.\S+', '', raw)   
cleaned = re.sub(r'@\w+', '', cleaned)                 
cleaned = re.sub(r'#\w+', '', cleaned)                 
cleaned = re.sub(r"[^a-zA-Z0-9\s']", ' ', cleaned)  
cleaned = re.sub(r'\s+', ' ', cleaned).strip()
print(f"Cleaned: {cleaned}")
print("\n")




print("E. LOWERCASING & NORMALIZATION------------------------------")
raw = "  Hello   WORLD!!   #NLP  @researchers  AMAZING!!!    "
print(f"Raw:        '{raw}'")

norm = raw.lower().strip()
norm = re.sub(r'\s+', ' ', norm)

norm = re.sub(r'[#@]\w+', '', norm)
norm = re.sub(r"[^\w\s']", '', norm)
norm = re.sub(r'\s+', ' ', norm).strip()
print(f"Normalized: '{norm}'")
print("\n")



print("F. REGEX FOR TEXT CLEANING-------------------------------")
print("\n")
# F1 — Extract emails
para   = "Contact support@example.com or admin@company.org or john.doe@mail.co.uk. Invalid: @nodomain"
emails = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', para)
print(f"F1 Emails found : {emails}")
print("\n")
# F2 — Remove numbers
text = "In 2023, about 4.5 billion people used the internet. 5G runs above 24 GHz."
print(f"F2 No numbers   : {re.sub(r'\s+', ' ', re.sub(r'\d+\.?\d*', '', text)).strip()}")
print("\n")
# F3 — Collapse spaces
messy = "This   has   too    many   spaces  and\t\ttabs."
print(f"F3 Collapsed    : '{re.sub(r'[ \t]+', ' ', messy).strip()}'")