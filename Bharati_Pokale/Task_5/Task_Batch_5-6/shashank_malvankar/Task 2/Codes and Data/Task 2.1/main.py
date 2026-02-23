from utils.data_loader import load_dataset
from utils.preprocessing import clean_text

from utils.topic_utils import extract_topics
from utils.evaluation import calculate_perplexity, calculate_coherence

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from models.lda_model import train_lda
from models.nmf_model import train_nmf

from dashboard.visualize import save_lda_visualization


if __name__ == "__main__":

    print("Loading dataset...")
    df = load_dataset()

    print("Cleaning dataset...")
    df["clean"] = df["text"].apply(clean_text)

    print("Training LDA...")

    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)

    X_count = count_vectorizer.fit_transform(df["clean"])

    lda_model = train_lda(X_count)

    lda_topics = extract_topics(
        lda_model,
        count_vectorizer.get_feature_names_out()
    )

    lda_perplexity = calculate_perplexity(lda_model, X_count)

    lda_coherence = calculate_coherence(
        df["clean"],
        lda_topics
    )

    print("Training NMF...")

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

    X_tfidf = tfidf_vectorizer.fit_transform(df["clean"])

    nmf_model = train_nmf(X_tfidf)

    nmf_topics = extract_topics(
        nmf_model,
        tfidf_vectorizer.get_feature_names_out()
    )

    nmf_coherence = calculate_coherence(
        df["clean"],
        nmf_topics
    )

    save_lda_visualization(
    lda_model,
    X_count,
    count_vectorizer,
    lda_perplexity,
    lda_coherence,
    nmf_coherence
    )

    print("\nMODEL COMPARISON")
    print("----------------------------")

    print("LDA Perplexity:", lda_perplexity)

    print("LDA Coherence:", lda_coherence)

    print("NMF Coherence:", nmf_coherence)

    print("\nDashboard generated successfully.")
