import nltk
import pandas as pd
from nltk.corpus import movie_reviews

# Download dataset
nltk.download('movie_reviews')

data = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        review = movie_reviews.raw(fileid)
        data.append([review, category])

# Create DataFrame
df = pd.DataFrame(data, columns=["review", "sentiment"])

# Save as CSV
df.to_csv("movie_reviews_dataset.csv", index=False)

print("Dataset saved as movie_reviews_dataset.csv")
print("Total rows:", len(df))
