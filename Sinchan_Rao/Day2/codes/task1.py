import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import random

st.title("Topic Modeling Dashboard — LDA vs NMF (100 Docs Version)")

space = [
    "NASA launches new space mission",
    "Astronauts explore the moon and mars",
    "Space telescope studies galaxies",
    "Rocket technology advances space travel",
    "Scientists discover new exoplanets"
]

tech = [
    "Artificial intelligence and machine learning",
    "Python programming for data science",
    "Deep learning neural networks",
    "Software engineering and cloud computing",
    "AI models process big data"
]

sports = [
    "Football world cup tournament",
    "Cricket match and player performance",
    "Olympic games competition",
    "Teams train for championship",
    "Sports analytics and coaching"
]

finance = [
    "Stock market trading and investing",
    "Economic policy and inflation",
    "Banking and financial growth",
    "Crypto currency and blockchain",
    "Investment portfolio strategy"
]

health = [
    "Medical research and vaccines",
    "Doctors treat patients in hospitals",
    "Nutrition and healthy diet",
    "Mental health awareness",
    "Exercise improves fitness"
]

base_docs = space + tech + sports + finance + health

docs = [random.choice(base_docs) for _ in range(100)]


count_vec = CountVectorizer(max_features=500, stop_words="english")
tfidf_vec = TfidfVectorizer(max_features=500, stop_words="english")

count_matrix = count_vec.fit_transform(docs)
tfidf_matrix = tfidf_vec.fit_transform(docs)

n_topics = st.slider("Number of Topics", 2, 8, 5)


lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=6,
    learning_method="online",
    random_state=42
)
lda.fit(count_matrix)

nmf = NMF(
    n_components=n_topics,
    max_iter=300,
    random_state=42
)
nmf.fit(tfidf_matrix)


def show_topics(model, features, n_top=7):
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


st.subheader("Model Metrics")
st.write("LDA Perplexity:", round(lda.perplexity(count_matrix), 2))
st.write("NMF Reconstruction Error:", round(nmf.reconstruction_err_, 2))

st.success("Task 1 complete — using ~100 documents.")

