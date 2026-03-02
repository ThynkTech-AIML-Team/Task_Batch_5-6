import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
news_articles = [
    "Apple unveils new iPhone with better camera and longer battery life.",
    "Microsoft announces Windows 12 release date and new features.",
    "Tesla launches new electric car model with advanced autopilot.",
    "Apple reports record quarterly profits due to strong iPhone sales.",
    "Scientists discover water on Mars, raising hopes for life.",
    "SpaceX plans first civilian trip to orbit next year.",
    "Google introduces AI-powered search features for better results.",
    "NASA studies climate change impact on Earth's ice sheets.",
    "Amazon expands its delivery network to 10 more countries.",
    "Researchers develop new AI tool to detect cancer early."
]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(news_articles)
def find_similar_articles(input_text, tfidf_matrix, articles, top_n=3):
    input_vec = vectorizer.transform([input_text])
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    print("\nTop similar articles:")
    for i in top_indices:
        print(f"- {articles[i]} (Score: {cosine_sim[i]:.3f})")

new_article = "Apple releases new smartphone with improved battery and camera."
find_similar_articles(new_article, tfidf_matrix, news_articles)
