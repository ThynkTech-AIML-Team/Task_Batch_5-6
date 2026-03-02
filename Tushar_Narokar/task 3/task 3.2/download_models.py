import gensim.downloader as api
import os

print("Starting downloads...")
print("Downloading glove-wiki-gigaword-100 (~128 MB)...")
api.load('glove-wiki-gigaword-100')
print("GloVe downloaded.")

print("Downloading word2vec-google-news-300 (~1.7 GB)...")
# Note: This might take a while.
api.load('word2vec-google-news-300')
print("Word2Vec downloaded.")
