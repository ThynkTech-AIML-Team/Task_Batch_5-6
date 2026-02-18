import numpy as np

def load_glove(path):

    embeddings = {}

    with open(path, encoding="utf-8") as f:

        for line in f:

            values = line.split()

            word = values[0]

            vector = np.asarray(
                values[1:],
                dtype="float32"
            )

            embeddings[word] = vector

    print("Loaded GloVe:", len(embeddings))

    return embeddings