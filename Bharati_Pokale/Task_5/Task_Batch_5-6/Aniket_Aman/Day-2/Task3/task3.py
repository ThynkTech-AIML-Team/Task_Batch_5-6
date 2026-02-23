from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NewsRecommender:
    def __init__(self):
        print("1. Loading News Dataset... (this may take a moment)")
        # We fetch a subset of categories to keep it fast, or remove 'categories' to get all 20.
        cats = ['sci.space', 'rec.autos', 'comp.graphics', 'rec.sport.baseball', 'sci.med']
        self.dataset = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))
        
        print(f"   -> Loaded {len(self.dataset.data)} articles.")

        print("2. Building Search Index (TF-IDF)...")
        # Convert text to vectors (numbers)
        # stop_words='english' removes common words like "the", "is", "and"
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.dataset.data)
        
        print("   -> Index built successfully!")

    def find_similar(self, user_query):
        """
        Takes a user input, transforms it into the same vector space,
        and finds the top 3 most similar articles.
        """
        # 1. Vectorize the user's query
        query_vec = self.vectorizer.transform([user_query])
        
        # 2. Calculate Cosine Similarity between query and ALL articles
        # Result is an array of scores between 0 (no match) and 1 (perfect match)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 3. Get indices of top 3 matches (sorted by score descending)
        # argsort sorts ascending, so we take the last 3 and reverse them
        top_indices = similarities.argsort()[-3:][::-1]
        
        return top_indices, similarities

# --- Main Application ---
if __name__ == "__main__":
    app = NewsRecommender()
    
    print("\n" + "="*50)
    print(" NEWS SIMILARITY SEARCH ENGINE ")
    print("="*50)
    
    while True:
        user_input = input("\nEnter a news headline or topic (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
            
        indices, scores = app.find_similar(user_input)
        
        print(f"\nTop 3 Articles for: '{user_input}'")
        for i, idx in enumerate(indices):
            score = scores[idx]
            article_snippet = app.dataset.data[idx][:200].replace('\n', ' ') # First 200 chars
            category = app.dataset.target_names[app.dataset.target[idx]]
            
            print(f"\n{i+1}. [Score: {score:.4f}] [Category: {category}]")
            print(f"   Preview: \"{article_snippet}...\"")