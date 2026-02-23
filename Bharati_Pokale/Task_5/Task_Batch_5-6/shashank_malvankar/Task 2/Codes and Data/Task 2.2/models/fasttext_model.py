from gensim.models import FastText
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups

def train_fasttext():

    dataset = fetch_20newsgroups(
        remove=('headers', 'footers', 'quotes')
    )

    sentences = [
        simple_preprocess(text)
        for text in dataset.data
    ]

    model = FastText(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    print("FastText trained")

    return model
