from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups

def train_word2vec():

    dataset = fetch_20newsgroups(
        remove=('headers', 'footers', 'quotes')
    )

    sentences = [
        simple_preprocess(text)
        for text in dataset.data
    ]

    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    print("Word2Vec trained")

    return model
