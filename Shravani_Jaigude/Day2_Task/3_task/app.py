import streamlit as st
from news_similarity import load_news_data, build_vectorizer, get_top_k_similar

st.set_page_config(page_title="News Similarity Search", layout="wide")

st.title("End-to-End NLP App: News Similarity Search")
st.write("Enter a news/article text and get the **Top 3 most similar articles** from dataset.")

# Load dataset
@st.cache_data
def load_all():
    docs = load_news_data(n_samples=2500)
    vectorizer, X = build_vectorizer(docs)
    return docs, vectorizer, X

docs, vectorizer, X = load_all()

user_text = st.text_area("Enter News / Article Text:", height=200)

if st.button("Find Similar Articles"):
    if user_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        results = get_top_k_similar(user_text, vectorizer, X, docs, k=3)

        st.subheader("Top 3 Similar Articles")
        for i, res in enumerate(results, start=1):
            st.markdown(f"### Result {i}")
            st.write("**Similarity Score:**", round(res["similarity"], 4))
            st.write(res["text"][:800] + "...")
            st.divider()
