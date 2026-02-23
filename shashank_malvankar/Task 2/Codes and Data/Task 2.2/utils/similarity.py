import numpy as np

def cosine_similarity(v1, v2):

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )


def similarity_glove(w1, w2, glove):

    if w1 not in glove or w2 not in glove:
        return "OOV"

    return cosine_similarity(
        glove[w1],
        glove[w2]
    )


def similarity_gensim(w1, w2, model):

    try:
        return model.wv.similarity(w1, w2)
    except:
        return "OOV"
