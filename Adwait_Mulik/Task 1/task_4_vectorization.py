import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def run_task_4():
    print("--- STARTING TASK 4: BoW & TF-IDF VECTORIZATION ---")
    
    # Sample corpus for vectorization
    corpus = [
        "NLP is part of Artificial Intelligence",
        "Artificial Intelligence is the future of technology",
        "NLP helps computers understand human language"
    ]

    # --- 1. BAG-OF-WORDS (BoW) ---
    vectorizer_bow = CountVectorizer()
    bow_sparse = vectorizer_bow.fit_transform(corpus)
    
    # We use a robust conversion that avoids Pylance iteration/attribute errors
    bow_dense = bow_sparse.toarray()  # type: ignore 
    
    df_bow = pd.DataFrame(
        data=bow_dense, 
        columns=vectorizer_bow.get_feature_names_out()
    )
    
    print("\n[1] Bag-of-Words (BoW) Matrix:")
    print(df_bow)

    # --- 2. TF-IDF ---
    vectorizer_tfidf = TfidfVectorizer()
    tfidf_sparse = vectorizer_tfidf.fit_transform(corpus)
    
    # Same conversion method for TF-IDF to ensure stability
    tfidf_dense = tfidf_sparse.toarray()  # type: ignore
    
    df_tfidf = pd.DataFrame(
        data=tfidf_dense, 
        columns=vectorizer_tfidf.get_feature_names_out()
    )
    
    print("\n[2] TF-IDF Matrix:")
    print(df_tfidf)

    # --- 3. COMPARE WORD IMPORTANCE ---
    word_to_check = 'intelligence'
    if word_to_check in df_tfidf.columns:
        print(f"\n[3] Importance Score for '{word_to_check}':")
        # Calculating the mean importance score for the selected word
        print(f"TF-IDF Mean Score: {df_tfidf[word_to_check].mean():.4f}")

    # Save final results for office GitHub repo
    df_tfidf.to_csv('outputs/task_4_tfidf_results.csv', index=False)
    print("\n[SUCCESS] Results saved in 'outputs/task_4_tfidf_results.csv'.")

if __name__ == "__main__":
    run_task_4()