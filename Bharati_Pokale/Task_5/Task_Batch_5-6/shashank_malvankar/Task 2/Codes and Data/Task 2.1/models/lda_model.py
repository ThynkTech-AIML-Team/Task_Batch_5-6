from sklearn.decomposition import LatentDirichletAllocation

def train_lda(X, n_topics=10):

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(X)

    return lda