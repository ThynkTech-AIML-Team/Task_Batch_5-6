import numpy as np
from sklearn.metrics import pairwise_distances

def lda_perplexity(lda_model, doc_term_matrix):
    return lda_model.perplexity(doc_term_matrix)

def coherence_score(model, feature_names, top_n=10):
    """
    Simplified coherence: cosine similarity among top words.
    """
    topics = model.components_
    coherences = []
    for topic in topics:
        top_indices = topic.argsort()[-top_n:]
        top_values = topic[top_indices]
        if len(top_values) <= 1:
            continue
        sim_matrix = 1 - pairwise_distances([top_values], metric='cosine')
        coherences.append(np.mean(sim_matrix))
    return np.mean(coherences)

def display_topics(model, feature_names, n_top_words=10):
    topics_dict = {}
    for idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics_dict[f"Topic {idx+1}"] = ", ".join(top_features)
    return topics_dict
