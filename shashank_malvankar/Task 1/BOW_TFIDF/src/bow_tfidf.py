import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "data", "input_sentences.txt")

OUTPUT_BOW = os.path.join(BASE_DIR, "outputs", "bow_results.txt")

OUTPUT_TFIDF = os.path.join(BASE_DIR, "outputs", "tfidf_results.txt")

def load_sentences():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        sentences = file.readlines()

    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def bag_of_words(sentences):

    vectorizer = CountVectorizer()

    matrix = vectorizer.fit_transform(sentences)

    words = vectorizer.get_feature_names_out()

    matrix_array = matrix.toarray()

    return words, matrix_array

def tfidf(sentences):

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(sentences)

    words = vectorizer.get_feature_names_out()

    matrix_array = matrix.toarray()

    return words, matrix_array

def save_bow(words, matrix):

    os.makedirs(os.path.dirname(OUTPUT_BOW), exist_ok=True)

    with open(OUTPUT_BOW, "w", encoding="utf-8") as f:

        f.write("BAG OF WORDS MATRIX\n")
        f.write("===================\n\n")

        f.write("FEATURE WORDS:\n")
        f.write(str(list(words)))

        f.write("\n\nMATRIX:\n")

        for row in matrix:
            f.write(str(row) + "\n")

def save_tfidf(words, matrix):

    os.makedirs(os.path.dirname(OUTPUT_TFIDF), exist_ok=True)

    with open(OUTPUT_TFIDF, "w", encoding="utf-8") as f:

        f.write("TF-IDF MATRIX\n")
        f.write("=============\n\n")

        f.write("FEATURE WORDS:\n")
        f.write(str(list(words)))

        f.write("\n\nMATRIX:\n")

        for row in matrix:
            f.write(str(row) + "\n")

def main():

    print("Loading sentences...")

    sentences = load_sentences()

    print("Creating Bag of Words...")

    bow_words, bow_matrix = bag_of_words(sentences)

    save_bow(bow_words, bow_matrix)

    print("Creating TF-IDF...")

    tfidf_words, tfidf_matrix = tfidf(sentences)

    save_tfidf(tfidf_words, tfidf_matrix)

    print("Task 4 completed. Check outputs folder.")

if __name__ == "__main__":
    main()
