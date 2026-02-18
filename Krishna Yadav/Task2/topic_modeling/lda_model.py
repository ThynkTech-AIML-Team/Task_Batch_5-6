print("LDA Topic Modeling Started...")

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import preprocess_documents
from dataset_loader import load_data

import pyLDAvis
import pyLDAvis.lda_model


# -------------------------
# Load + preprocess
# -------------------------
print("Loading dataset...")
docs = load_data()

print("Preprocessing documents...")
processed_docs = preprocess_documents(docs[:500])  # small batch first

# -------------------------
# Count Vectorizer (LDA prefers counts)
# -------------------------
print("Creating Count Vectorizer matrix...")

vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=2000
)

X = vectorizer.fit_transform(processed_docs)

# -------------------------
# Train LDA
# -------------------------
NUM_TOPICS = 5

print("Training LDA model...")

lda_model = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    random_state=42
)

lda_model.fit(X)

# -------------------------
# Show Topics
# -------------------------
feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("\nTop Words Per Topic:")
display_topics(lda_model, feature_names)

# -------------------------
# Create Interactive Dashboard
# -------------------------
print("\nGenerating pyLDAvis dashboard...")

vis = pyLDAvis.lda_model.prepare(lda_model, X, vectorizer)


pyLDAvis.save_html(vis, "lda_dashboard.html")

print("\nDashboard saved as lda_dashboard.html")
