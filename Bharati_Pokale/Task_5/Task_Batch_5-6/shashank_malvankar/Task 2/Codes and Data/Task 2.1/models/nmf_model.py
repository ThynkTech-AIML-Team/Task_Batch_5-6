from sklearn.decomposition import NMF

def train_nmf(X, n_topics=10):

    nmf = NMF(
        n_components=n_topics,
        random_state=42
    )

    nmf.fit(X)

    return nmf
