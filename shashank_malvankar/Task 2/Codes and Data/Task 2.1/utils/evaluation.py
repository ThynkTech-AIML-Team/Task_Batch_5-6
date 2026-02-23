from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


def calculate_perplexity(model, X):
    return model.perplexity(X)


def calculate_coherence(texts, topics):

    tokenized_texts = [text.split() for text in texts]

    dictionary = Dictionary(tokenized_texts)

    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    return coherence_model.get_coherence()
