# Step 1
from sklearn.datasets import fetch_20newsgroups

print("Loading dataset...")

dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

documents = dataset.data

print("Total documents:", len(documents))

print("\nSample document:\n")
print(documents[0][:300])

# Step 2

from sklearn.feature_extraction.text import TfidfVectorizer

print("\nCreating TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000
)

X = vectorizer.fit_transform(documents)

print("TF-IDF shape:", X.shape)

# Step 3

from sklearn.decomposition import LatentDirichletAllocation

print("\nTraining LDA model...")

lda = LatentDirichletAllocation(
    n_components=5,     
    random_state=42
)

lda.fit(X)

print("LDA model trained successfully")

# Step 4

print("\nTopics from LDA:")

feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    
    print(f"\nTopic {topic_idx + 1}:")
    
    top_words = [
        feature_names[i]
        for i in topic.argsort()[:-11:-1]
    ]
    
    print(", ".join(top_words))

# Step 5: Train NMF Model

from sklearn.decomposition import NMF

print("\nTraining NMF model...")

nmf = NMF(
    n_components=5,
    random_state=42
)

nmf.fit(X)

print("NMF model trained successfully")


# Display NMF Topics

print("\nTopics from NMF:")

for topic_idx, topic in enumerate(nmf.components_):
    
    print(f"\nTopic {topic_idx + 1}:")
    
    top_words = [
        feature_names[i]
        for i in topic.argsort()[:-11:-1]
    ]
    
    print(", ".join(top_words))

# Step 6: Create visualization dashboard (WORKING VERSION)

import pyLDAvis
import pyLDAvis.lda_model

print("\nCreating LDA visualization dashboard...")

lda_vis = pyLDAvis.lda_model.prepare(
    lda,
    X,
    vectorizer,
    mds='tsne'
)

# Save dashboard
pyLDAvis.save_html(
    lda_vis,
    "lda_dashboard.html"
)

print("Dashboard saved as lda_dashboard.html")

# Step 7: Compare models

print("\nModel Comparison:")

lda_perplexity = lda.perplexity(X)

print("LDA Perplexity:", lda_perplexity)

nmf_error = nmf.reconstruction_err_

print("NMF Reconstruction Error:", nmf_error)
