from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

news_articles = [

    "Apple releases new iPhone with advanced camera",

    "Samsung launches new Galaxy smartphone",

    "Google announces new Android version",

    "Microsoft releases Windows update",

    "Tesla introduces new electric car",

    "Amazon launches new online shopping features",

    "Facebook introduces new social media tools"

]


print("News Similarity Search System Ready")

vectorizer = TfidfVectorizer()

news_vectors = vectorizer.fit_transform(news_articles)


query = input("\nEnter your news article: ")


query_vector = vectorizer.transform([query])
similarity = cosine_similarity(query_vector, news_vectors)
top_indices = similarity[0].argsort()[-3:][::-1]
print("\nTop 3 similar news articles:\n")

for i in top_indices:
    
    print(news_articles[i])
