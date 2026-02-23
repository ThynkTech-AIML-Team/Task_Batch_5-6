import numpy as np

avg_tfidf = np.mean(X_tfidf.toarray(), axis=0)

importance_df = pd.DataFrame({
    "Word": tfidf_vectorizer.get_feature_names_out(),
    "Average TF-IDF Score": avg_tfidf
}).sort_values(by="Average TF-IDF Score", ascending=False)

print("\nWord Importance (Highest to Lowest):")
print(importance_df)