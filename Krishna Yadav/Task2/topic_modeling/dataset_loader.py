print("Script Started...")

from sklearn.datasets import fetch_20newsgroups

def load_data():
    dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    return dataset.data

if __name__ == "__main__":
    docs = load_data()
    print("Total documents:", len(docs))
    print(docs[0][:500])
