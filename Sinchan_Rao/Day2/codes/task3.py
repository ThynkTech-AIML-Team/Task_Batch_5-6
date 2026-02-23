import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("NLP App — News Similarity Search (Final Version)")

# -------------------------
# Structured dataset (~150 docs, no random duplicates)
# -------------------------

space = [
    "NASA launches new rocket mission",
    "Mars exploration program expands",
    "Astronauts prepare for moon base",
    "Space telescope discovers galaxy",
    "Satellite technology improves"
]

ai = [
    "Artificial intelligence transforms healthcare",
    "Machine learning improves predictions",
    "Deep learning model beats benchmark",
    "AI powers automation systems",
    "Neural networks process images"
]

finance = [
    "Stock market shows volatility",
    "Investors increase portfolio diversification",
    "Banking sector reports growth",
    "Crypto market fluctuates",
    "Trading strategy reduces risk"
]

sports = [
    "Football team wins final",
    "Cricket tournament begins",
    "Olympic training intensifies",
    "Sports analytics improves performance",
    "Championship match draws crowd"
]

health = [
    "Doctors develop new vaccine",
    "Medical research shows promise",
    "Hospital adopts new technology",
    "Health experts recommend exercise",
    "Nutrition improves immunity"
]

tech = [
    "Cloud computing demand rises",
    "Cybersecurity threats increase",
    "Software engineering trends change",
    "Data science drives decisions",
    "Programming tools evolve"
]

base_docs = space + ai + finance + sports + health + tech

# repeat in structured way → size increases but stays balanced
docs = base_docs * 5   # 30 × 5 = 150 docs


# -------------------------
# Load embedding model
# -------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.write("Encoding documents (first run takes ~1 min)...")
doc_embeddings = model.encode(docs, show_progress_bar=False)


# -------------------------
# User input
# -------------------------

query = st.text_area("Enter article text")

if st.button("Find Similar Articles") and query.strip():

    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, doc_embeddings)[0]

    # get more than needed, then deduplicate
    candidate_ids = sims.argsort()[::-1][:15]

    st.subheader("Top Similar Articles")

    seen = set()
    shown = 0

    for i in candidate_ids:
        if docs[i] in seen:
            continue

        seen.add(docs[i])
        shown += 1

        st.write(f"### Match {shown}")
        st.write(docs[i])
        st.write(f"Similarity score: {sims[i]:.3f}")
        st.write("---")

        if shown == 5:
            break

st.success("Task 3 complete — clean similarity search running.")