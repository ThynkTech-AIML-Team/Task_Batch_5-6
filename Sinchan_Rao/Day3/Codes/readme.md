**Task 1 – Text Classification**

In this task, a text classification model was developed to identify spam messages using the SMS Spam Dataset. The goal was to compare different machine learning algorithms and evaluate their performance in detecting spam and non-spam (ham) messages.

Models Applied:

Naive Bayes

Logistic Regression

SVM

Results:

Naive Bayes Accuracy: 98%

Logistic Regression Accuracy: 97%

SVM Accuracy: 98%

All three models performed very well, with Naive Bayes and SVM achieving the highest accuracy.

Tools:

Python

scikit-learn

NLTK



**Task 2 – Word Embedding**

Objective:

The goal of this task was to understand how word embeddings capture semantic meaning and relationships between words using pre-trained models.

Implementation:

Loaded the pre-trained Word2Vec model (word2vec-google-news-300) using gensim.

Retrieved similar words for a given term.

Visualized 10–20 selected words in 2D space using Principal Component Analysis (PCA) to observe clustering of semantically related words.

Results:

The model successfully identified semantically similar words.

Word analogy operations produced meaningful results.

PCA visualization showed that related words appear close to each other in vector space.

Tools:

Python

gensim

NumPy

scikit-learn

Matplotlib




**Task 3- Option C – Rule-Based FAQ Chatbot**

Objective:

To develop a simple rule-based FAQ chatbot that answers user queries by matching them with the most relevant predefined question using text similarity techniques.

Implementation:

Created 15 predefined question–answer pairs stored in a Python dictionary.

Extracted all questions and converted them into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency) with TfidfVectorizer from scikit-learn.

Transformed the user’s query into a TF-IDF vector using the same vectorizer.

Computed Cosine Similarity between the user query vector and all stored question vectors.

Identified the question with the highest similarity score using numpy.argmax().

If the similarity score was above a defined threshold (0.3), the corresponding answer was returned.

If the score was below the threshold, the chatbot responded with a fallback message:
“Sorry, I couldn't understand your question.”

Implemented a simple loop interface where the user can continuously ask questions until typing "exit" to stop the chatbot.

Tools:

Python

NumPy

scikit-learn (TF-IDF Vectorizer, Cosine Similarity)

Outcome

The chatbot successfully matches user queries with the closest predefined FAQ using TF-IDF and cosine similarity. This demonstrates how a rule-based chatbot can be built using basic NLP techniques without deep learning models.
