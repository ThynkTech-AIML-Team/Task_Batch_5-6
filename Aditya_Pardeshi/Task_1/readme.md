## 1. Introduction to NLP 
Task: 
Write a short Python script using any open dataset (e.g., movie reviews, tweets) to 
demonstrate one real-world NLP application: 
• Sentiment analysis (positive/negative) 
• Text summarization (using a library like sumy or gensim) 
## 2. Text Data Basics 
Task: 
Write a Python function that takes a paragraph and outputs: 
• List of sentences 
• List of tokens (words) 
• Count of tokens, sentences, paragraphs 
## 3. Text Preprocessing 
### A. Tokenization 
Task: 
Tokenize a text document using: 
NLTK word_tokenizer,spaCy tokenizer 
Compare outputs. 
### B. Stopwords Removal 
Task: Remove stopwords from a text and print before vs. after. 
### C. Lemmatization & Stemming 
Task: 
Apply PorterStemmer and WordNetLemmatizer on a text chapter. 
Show differences (e.g., “studies” → “studi” vs. “study”). 
### D. Handling punctuation, special characters, emojis 
Task:  
Clean text containing #hashtags, @mentions, !!!,          
### E. Lowercasing & Normalization 
using regex + emoji library. 
Task: Convert mixed-case text into lowercase and normalize spacing. 
### F. Regex for text cleaning 
Task: 
Extract all emails from a paragraph. 
Remove numbers from text. 
Replace multiple spaces with one. 
## 4. Bag-of-Words (BoW), TF-IDF 
Task: 
Using scikit-learn: 
Convert a list of sentences into a BoW matrix. 
Convert the same sentences into a TF-IDF matrix. 
Compare word importance scores. 
