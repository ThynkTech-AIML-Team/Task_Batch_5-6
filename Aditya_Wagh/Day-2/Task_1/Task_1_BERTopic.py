# BERTopic Topic Modeling

from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

print("Loading dataset...")

# Load dataset
dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes")
)

documents = dataset.data[:2000]   # limit for faster training

print("Total documents:", len(documents))


print("\nTraining BERTopic model...")

# Create BERTopic model
topic_model = BERTopic()

# Train model
topics, probs = topic_model.fit_transform(documents)

print("BERTopic model trained successfully")


# Show topics
print("\nTop Topics:\n")

topic_info = topic_model.get_topic_info()

print(topic_info.head())


# Show words per topic
print("\nTopic keywords:\n")

for topic in topic_info.Topic[1:6]:
    
    print(f"\nTopic {topic}:")
    
    words = topic_model.get_topic(topic)
    
    for word, score in words[:10]:
        
        print(word)


# Save visualization
print("\nSaving visualization...")

fig = topic_model.visualize_topics()

fig.write_html("bertopic_dashboard.html")

print("Visualization saved as bertopic_dashboard.html")
