print("NMF Topic Modeling Started...")

from sklearn.decomposition import NMF
from vectorizer import X, vectorizer

# Number of topics (you can change later)
NUM_TOPICS = 5

print("Training NMF model...")

nmf_model = NMF(
    n_components=NUM_TOPICS,
    random_state=42
)

nmf_model.fit(X)

feature_names = vectorizer.get_feature_names_out()


def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        print(
            " ".join(
                [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            )
        )


print("\nTop Words Per Topic:")
display_topics(nmf_model, feature_names)
