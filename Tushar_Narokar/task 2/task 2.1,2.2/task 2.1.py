
import sys, warnings, ssl
warnings.filterwarnings("ignore")

try:
    _default_https_context = ssl._create_default_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

import pyLDAvis
import pyLDAvis.lda_model


N_TOPICS      = 6
N_TOP_WORDS   = 10
N_DOCS        = 2000
RANDOM_STATE  = 42

CATEGORIES = [
    "sci.space", "sci.med", "rec.sport.baseball",
    "rec.autos", "talk.politics.guns", "comp.graphics",
]


def load_data():
    print("Loading 20 Newsgroups dataset...")
    data = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    docs = [d for d in data.data if len(d.strip()) > 50][:N_DOCS]
    print(f"  Loaded {len(docs)} documents across {len(CATEGORIES)} categories")
    return docs


def preprocess(docs):
    import re
    clean = []
    for doc in docs:
        doc = doc.lower()
        doc = re.sub(r"[^a-z\s]", " ", doc)
        doc = re.sub(r"\s+", " ", doc).strip()
        clean.append(doc)
    return clean


def vectorize(docs):
    cv = CountVectorizer(max_df=0.90, min_df=5, max_features=5000,
                         stop_words="english")
    dtm_count = cv.fit_transform(docs)

    tv = TfidfVectorizer(max_df=0.90, min_df=5, max_features=5000,
                         stop_words="english")
    dtm_tfidf = tv.fit_transform(docs)

    print(f"  Vocabulary size: {len(cv.get_feature_names_out())} terms")
    print(f"  Document-term matrix: {dtm_count.shape}")
    return cv, dtm_count, tv, dtm_tfidf


def train_lda(dtm, n_topics=N_TOPICS):
    print(f"\nTraining LDA ({n_topics} topics)...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="online",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lda.fit(dtm)
    print("  Done.")
    return lda


def train_nmf(dtm_tfidf, n_topics=N_TOPICS):
    print(f"Training NMF ({n_topics} topics)...")
    nmf = NMF(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=400,
    )
    nmf.fit(dtm_tfidf)
    print("  Done.")
    return nmf


def print_topics(model, vectorizer, model_name="Model", n_words=N_TOP_WORDS):
    feature_names = vectorizer.get_feature_names_out()
    bar = "─" * 55
    print(f"\n{bar}")
    print(f"  {model_name} — Top {n_words} words per topic")
    print(f"{bar}")
    for i, topic in enumerate(model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-n_words-1:-1]]
        print(f"  Topic {i+1:>2}: {', '.join(top_words)}")
    print(bar)


def coherence_score(model, vectorizer, docs_clean, model_name):
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic in model.components_:
        top_idx   = topic.argsort()[:-N_TOP_WORDS-1:-1]
        top_words = [feature_names[i] for i in top_idx]
        topics.append(top_words)

    tokenized = [simple_preprocess(doc) for doc in docs_clean]
    dictionary = corpora.Dictionary(tokenized)
    corpus     = [dictionary.doc2bow(tok) for tok in tokenized]

    cm = CoherenceModel(topics=topics, texts=tokenized,
                        dictionary=dictionary, coherence="c_v")
    score = cm.get_coherence()
    print(f"  {model_name} Coherence (C_v): {score:.4f}")
    return score


def perplexity_score(lda_model, dtm):
    score = lda_model.perplexity(dtm)
    print(f"  LDA Perplexity:              {score:.2f}  (lower = better)")
    return score


def plot_comparison(lda_coh, nmf_coh, lda_perp):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("LDA vs NMF — Model Comparison", fontsize=13, y=1.02)

    axes[0].bar(["LDA", "NMF"], [lda_coh, nmf_coh],
                color=["#5C9BE0", "#5CBF7A"], edgecolor="white", width=0.4)
    axes[0].set_title("Coherence Score (C_v)\nhigher = better", fontsize=10)
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([lda_coh, nmf_coh]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(["LDA"], [lda_perp], color=["#E05C5C"],
                edgecolor="white", width=0.3)
    axes[1].set_title("Perplexity (LDA only)\nlower = better", fontsize=10)
    axes[1].text(0, lda_perp + 10, f"{lda_perp:.1f}", ha="center", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved → model_comparison.png")
    plt.close()


def plot_topic_heatmap(model, vectorizer, model_name, save_as):
    feature_names = vectorizer.get_feature_names_out()
    n_topics = model.components_.shape[0]

    top_words_per_topic = []
    all_top = set()
    for topic in model.components_:
        top = [feature_names[j] for j in topic.argsort()[:-12:-1]]
        top_words_per_topic.append(top)
        all_top.update(top)

    all_top  = sorted(all_top)
    word_idx = {w: i for i, w in enumerate(all_top)}

    matrix = np.zeros((n_topics, len(all_top)))
    for t, topic in enumerate(model.components_):
        norm = topic / (topic.sum() + 1e-9)
        for w in top_words_per_topic[t]:
            matrix[t, word_idx[w]] = norm[list(feature_names).index(w)]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(all_top)))
    ax.set_xticklabels(all_top, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels([f"Topic {i+1}" for i in range(n_topics)], fontsize=9)
    ax.set_title(f"{model_name} — Topic-Word Weight Heatmap", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_as}")
    plt.close()


def save_pyldavis(lda_model, dtm, cv):
    print("\nGenerating pyLDAvis dashboard...")
    panel = pyLDAvis.lda_model.prepare(lda_model, dtm, cv, mds="tsne")
    pyLDAvis.save_html(panel, "lda_dashboard.html")
    print("  Saved → lda_dashboard.html  (open in browser)")


def run_bertopic(docs_clean):
    try:
        from bertopic import BERTopic
    except ImportError:
        print("\n  BERTopic not installed — run:  pip install bertopic")
        return

    print("\nRunning BERTopic (transformer-based)...")
    print("  This may take a few minutes on first run (downloads sentence-transformers)...")

    model = BERTopic(
        nr_topics=N_TOPICS,
        verbose=False,
        calculate_probabilities=False,
    )
    topics, _ = model.fit_transform(docs_clean[:500])

    print(f"\n  BERTopic — Top words per topic:")
    print("  " + "─" * 50)
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = [w for w, _ in model.get_topic(topic_id)[:8]]
        print(f"  Topic {topic_id+1:>2}: {', '.join(words)}")

    fig = model.visualize_barchart(top_n_topics=N_TOPICS, n_words=8)
    fig.write_image("bertopic_barchart.png")
    print("  Saved → bertopic_barchart.png")

    return model


def main():
    run_bert = "--no-bert" not in sys.argv

    print("\n=== 1. Loading Data ===")
    docs = load_data()

    print("\n=== 2. Preprocessing ===")
    docs_clean = preprocess(docs)

    print("\n=== 3. Vectorizing ===")
    cv, dtm_count, tv, dtm_tfidf = vectorize(docs_clean)

    print("\n=== 4 & 5. Training LDA & NMF ===")
    lda = train_lda(dtm_count)
    nmf = train_nmf(dtm_tfidf)

    print("\n=== 6. Discovered Topics ===")
    print_topics(lda, cv,  model_name="LDA")
    print_topics(nmf, tv,  model_name="NMF")

    print("\n=== 7. Coherence & Perplexity ===")
    lda_coh  = coherence_score(lda, cv, docs_clean, "LDA")
    nmf_coh  = coherence_score(nmf, tv, docs_clean, "NMF")
    lda_perp = perplexity_score(lda, dtm_count)

    print("\n=== 8. Comparison Chart ===")
    plot_comparison(lda_coh, nmf_coh, lda_perp)

    print("\n=== 9. Topic-Word Heatmaps ===")
    plot_topic_heatmap(lda, cv, "LDA", save_as="lda_heatmap.png")
    plot_topic_heatmap(nmf, tv, "NMF", save_as="nmf_heatmap.png")

    print("\n=== 10. pyLDAvis Dashboard ===")
    save_pyldavis(lda, dtm_count, cv)

    if run_bert:
        print("\n=== 11. BERTopic (Bonus) ===")
        run_bertopic(docs_clean)
    else:
        print("\n  Skipping BERTopic (--no-bert flag)")

    print("\n" + "─" * 55)
    print("  All outputs saved:")
    print("    lda_dashboard.html   ← open in browser (pyLDAvis)")
    print("    model_comparison.png ← LDA vs NMF scores")
    print("    lda_heatmap.png      ← LDA topic-word weights")
    print("    nmf_heatmap.png      ← NMF topic-word weights")
    if run_bert:
        print("    bertopic_barchart.png← BERTopic top words")
    print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
