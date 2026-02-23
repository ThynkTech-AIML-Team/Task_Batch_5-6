from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def load_dataset():

    dataset = fetch_20newsgroups(
        remove=('headers', 'footers', 'quotes')
    )

    df = pd.DataFrame({
        "text": dataset.data
    })

    df = df.dropna()

    return df
