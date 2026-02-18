def analogy_gensim(w1, w2, w3, model):

    try:
        result = model.wv.most_similar(
            positive=[w2, w3],
            negative=[w1],
            topn=1
        )

        return result[0][0]

    except:
        return "OOV"
