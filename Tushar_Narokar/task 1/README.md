# NLP & Text Processing Tasks

This repository contains a collection of four tasks focused on Natural Language Processing (NLP), ranging from fundamental text analysis to machine learning-based sentiment classification and text representation techniques.

##  Project Overview

| File | Description | Key Technologies |
|------|-------------|------------------|
| `task1.py` | Sentiment Analysis on Twitter data. | NLTK, Scikit-Learn (Logistic Regression) |
| `task2.py` | Basic text counting and tokenization. | NLTK (sent_tokenize, word_tokenize) |
| `task3.py` | Comprehensive text preprocessing workflow. | Regex, NLTK (Stemming, Lemmatization) |
| `task4.py` | Comparison of BoW and TF-IDF representations. | Pandas, Scikit-Learn (Cosine Similarity) |

---

##  Task Details

### Task 1: Sentiment Analysis
Objective: Classify tweets into sentiment categories (positive/negative).
- **Dataset:** NLTK `twitter_samples`.
- **Methodology:** TF-IDF calculation followed by a Logistic Regression model.
- **Output:** Classification report showing Precision, Recall, and Accuracy.

### Task 2: Text Analysis & Counting
Objective: Extract structural information from raw text.
- **Metrics:** Counts for paragraphs, sentences, and tokens.
- **Process:** Utilizes NLTK's tokenizers to break down complex strings into manageable units.

### Task 3: NLP Preprocessing Pipeline
Objective: Demonstrate best practices for cleaning raw text data.
- **Features:**
    - Tokenization (Comparing NLTK with custom regex methods).
    - Stopword removal for noise reduction.
    - Comparison of Stemming vs. Lemmatization.
    - Cleaning of URLs, @mentions, #hashtags, and special characters.
    - Regex-based extraction of emails and numeric values.

### Task 4: Text Representation Comparison
Objective: Analyze how different vectorization methods affect text similarity.
- **Comparison:** Bag of Words (BoW) vs. TF-IDF.
- **Metrics:** Global word importance and Cosine Similarity between multiple sentences.
- **Highlights:** Identifies top 3 important words per sentence based on TF-IDF scoring.

---

## Summary Table

| Task | Goal | Key Output |
| :--- | :--- | :--- |
| **Task 1** | Sentiment Classification | Accuracy & Prediction |
| **Task 2** | Structural Analysis | Token/Sentence Counts |
| **Task 3** | Text Cleaning | Cleaned & Normalized Text |
| **Task 4** | Feature Engineering | Similarity Matrices |

---

## Execution Results
Visual outputs of the tasks are stored in the `task outpust` folder:
- Sentiment classification metrics (`task1.png`).
- Tokenization and count outputs (`task2.png`).
- Preprocessing steps and regex extraction results (`task3.1.png`, `task3.2.png`).
- BoW/TF-IDF matrices and similarity scores (`task4.1.png`, `task4.2.png`, `task4.3.png`).

---

##  Setup and Usage

### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install nltk scikit-learn numpy pandas
```

### Download NLTK Data
Run the following commands to download necessary resources:
```python
import nltk
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running the Tasks
Execute any script individually to see the results:
```bash
python task1.py
python task2.py
python task3.py
python task4.py
```
