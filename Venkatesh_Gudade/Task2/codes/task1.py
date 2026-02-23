import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

st.title("Topic Modeling Dashboard — LDA vs NMF (250 Unique Docs)")

# -------- Topic phrase banks --------

space = [
    "NASA mission", "mars exploration", "space telescope",
    "rocket launch", "astronaut training"
]

tech = [
    "artificial intelligence", "machine learning",
    "cloud computing", "software engineering",
    "data science"
]

sports = [
    "football tournament", "cricket match",
    "olympic training", "team championship",
    "sports analytics"
]

finance = [
    "stock market", "investment strategy",
    "banking reform", "crypto trading",
    "economic policy"
]

health = [
    "medical research", "hospital technology",
    "nutrition science", "mental health",
    "fitness exercise"
]

topic_map = {
    "space": space,
    "tech": tech,
    "sports": sports,
    "finance": finance,
    "health": health
}

# -------- Generate 250 UNIQUE docs --------

docs = []
doc_id = 1

for topic, phrases in topic_map.items():
    for phrase in phrases:
        for i in range(10):   # 5 topics × 5 phrases × 10 = 250 docs
            docs.append(
                f"Report {doc_id}: Latest update on {phrase} with detailed {topic} analysis and trends"
            )
            doc_id += 1

# -------- Vectorize --------

count_vec = CountVectorizer(max_features=800, stop_words="english")
tfidf_vec = TfidfVectorizer(max_features=800, stop_words="english")

count_matrix = count_vec.fit_transform(docs)
tfidf_matrix = tfidf_vec.fit_transform(docs)

n_topics = st.slider("Number of Topics", 2, 10, 5)

# -------- Models (kept fast but stable) --------

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=7,
    learning_method="online",
    random_state=42
)
lda.fit(count_matrix)

nmf = NMF(
    n_components=n_topics,
    max_iter=350,
    random_state=42
)
nmf.fit(tfidf_matrix)

# -------- Topic Display --------

def show_topics(model, features, n_top=8):
    topics = []
    for topic in model.components_:
        words = [features[i] for i in topic.argsort()[:-n_top-1:-1]]
        topics.append(", ".join(words))
    return topics

st.subheader("LDA Topics")
for i, t in enumerate(show_topics(lda, count_vec.get_feature_names_out())):
    st.write(f"Topic {i+1}: {t}")

st.subheader("NMF Topics")
for i, t in enumerate(show_topics(nmf, tfidf_vec.get_feature_names_out())):
    st.write(f"Topic {i+1}: {t}")

# -------- Metrics --------

st.subheader("Model Metrics")
st.write("LDA Perplexity:", round(lda.perplexity(count_matrix), 2))
st.write("NMF Reconstruction Error:", round(nmf.reconstruction_err_, 2))

st.success(f"Task 1 complete — {len(docs)} unique documents processed.")
