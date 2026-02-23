from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_documents
from dataset_loader import load_data

print("Loading dataset...")
docs = load_data()

print("Preprocessing documents...")
processed_docs = preprocess_documents(docs[:500])  # small batch for testing


print("Creating TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=2000
)

X = vectorizer.fit_transform(processed_docs)

print("\nTF-IDF Shape:", X.shape)
print("Sample Features:", vectorizer.get_feature_names_out()[:20])
