from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

sentences = [
    "NLP is powerful and useful",
    "NLP is used in machine learning",
    "Machine learning makes NLP powerful"
]

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(sentences)

bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

print("Bag-of-Words Matrix:")
print(bow_df)