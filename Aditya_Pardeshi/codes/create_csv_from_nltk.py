import nltk
import pandas as pd
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')

data = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        review_text = movie_reviews.raw(fileid)
        data.append([review_text, category])

# Create DataFrame
df = pd.DataFrame(data, columns=["review", "sentiment"])

# Save to CSV
df.to_csv("movie_reviews.csv", index=False)

print("CSV file created successfully!")
