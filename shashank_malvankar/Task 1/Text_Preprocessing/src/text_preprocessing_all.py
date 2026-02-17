import os
import re
import nltk
import spacy
import emoji

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "data", "input_text.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_text():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return file.read()

def task_A_tokenization(text):

    nltk_tokens = word_tokenize(text)

    spacy_tokens = [token.text for token in nlp(text)]

    output_file = os.path.join(OUTPUT_DIR, "01_tokenization.txt")

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("TASK 3A: TOKENIZATION COMPARISON\n")
        f.write("=================================\n\n")

        f.write("NLTK TOKENS:\n")
        f.write(str(nltk_tokens))

        f.write("\n\nSPACY TOKENS:\n")
        f.write(str(spacy_tokens))

        f.write("\n\nTOKEN COUNTS:\n")
        f.write(f"NLTK count: {len(nltk_tokens)}\n")
        f.write(f"spaCy count: {len(spacy_tokens)}\n")

    print("Task 3A completed")

def task_B_stopwords(text):

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    output_file = os.path.join(OUTPUT_DIR, "02_stopwords.txt")

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("TASK 3B: STOPWORD REMOVAL\n")
        f.write("=========================\n\n")

        f.write("ORIGINAL TOKENS:\n")
        f.write(str(tokens))

        f.write("\n\nAFTER STOPWORD REMOVAL:\n")
        f.write(str(filtered_tokens))

    print("Task 3B completed")

def task_C_stemming_lemmatization(text):

    tokens = word_tokenize(text)

    stemmer = PorterStemmer()

    lemmatizer = WordNetLemmatizer()

    stemmed = [stemmer.stem(word) for word in tokens]

    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    output_file = os.path.join(OUTPUT_DIR, "03_stemming_lemmatization.txt")

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("TASK 3C: STEMMING & LEMMATIZATION\n")
        f.write("==================================\n\n")

        f.write("ORIGINAL:\n")
        f.write(str(tokens))

        f.write("\n\nSTEMMED:\n")
        f.write(str(stemmed))

        f.write("\n\nLEMMATIZED:\n")
        f.write(str(lemmatized))

    print("Task 3C completed")

def task_D_cleaning(text):

    cleaned = re.sub(r'[@#]\w+', '', text)

    cleaned = emoji.replace_emoji(cleaned, replace='')

    cleaned = re.sub(r'[^\w\s]', '', cleaned)

    output_file = os.path.join(OUTPUT_DIR, "04_cleaning.txt")

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("TASK 3D: CLEANING TEXT\n")
        f.write("======================\n\n")

        f.write(cleaned)

    print("Task 3D completed")

def task_E_normalization(text):

    normalized = text.lower()

    normalized = re.sub(r'\s+', ' ', normalized)

    output_file = os.path.join(OUTPUT_DIR, "05_normalization.txt")

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("TASK 3E: LOWERCASE & NORMALIZATION\n")
        f.write("===================================\n\n")

        f.write(normalized)

    print("Task 3E completed")


def task_F_regex(text):

    emails = re.findall(r'\S+@\S+', text)

    no_numbers = re.sub(r'\d+', '', text)

    single_space = re.sub(r'\s+', ' ', text)

    output_file = os.path.join(OUTPUT_DIR, "06_regex.txt")

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("TASK 3F: REGEX OPERATIONS\n")
        f.write("=========================\n\n")

        f.write("EMAILS FOUND:\n")
        f.write(str(emails))

        f.write("\n\nTEXT WITHOUT NUMBERS:\n")
        f.write(no_numbers)

        f.write("\n\nTEXT WITH NORMALIZED SPACING:\n")
        f.write(single_space)

    print("Task 3F completed")

def main():

    print("Loading input text...\n")

    text = load_text()

    task_A_tokenization(text)

    task_B_stopwords(text)

    task_C_stemming_lemmatization(text)

    task_D_cleaning(text)

    task_E_normalization(text)

    task_F_regex(text)

    print("\nALL TASK 3 SUBTASKS COMPLETED")


if __name__ == "__main__":
    main()
