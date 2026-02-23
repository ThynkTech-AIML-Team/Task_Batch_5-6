**Dataset:**
For the Fake News Detection task, I used two CSV files: `Fake.csv` (containing fake news articles) and `True.csv` (containing real news articles). These datasets include news titles and text, which are combined and cleaned for analysis.

**Model Used:**
I experimented with two main models: Logistic Regression and Naive Bayes. Both models were trained on TF-IDF vectorized text data to classify news as real or fake.

**Result:**
Both models performed well, with Logistic Regression achieving high accuracy (typically above 95% on the test set). The notebooks also show confusion matrices and highlight the most important words for each class, making the results easy to interpret.


# NLP Mini Projects

### 1. Basic Text Classification
**File:** `1_Basic_Text_Classification.ipynb`

 This notebook walks you through loading data, cleaning it up, turning words into numbers (using TF-IDF), and training a simple model to classify text. It’s a great intro to the basics of NLP and machine learning.

### 2. Word Embedding Mini Task
**File:** `2_Word_Embedding_Mini_Task.ipynb`

Ever wondered how computers “understand” word meanings? Here, you’ll play with powerful pretrained word embeddings like Word2Vec and GloVe. You’ll find similar words, solve analogies, and even visualize word relationships in 2D using PCA. It’s a cool way to see language in action!

### 3. Fake News Detection System
**Folder:** `Fake_news_detection/`
  - **Notebook:** `3_Fake_News_Detection_System.ipynb`
  - **Data:** `Fake.csv`, `True.csv`
      ()
Spotting fake news! This project guides you through building a full pipeline—from cleaning and vectorizing news articles, to training and comparing models, to interpreting results. You’ll see which words matter most and visualize how well your models perform.

---

## Project Structure

```
1_Basic_Text_Classification.ipynb
2_Word_Embedding_Mini_Task.ipynb
Fake_news_detection/
    3_Fake_News_Detection_System.ipynb
    Fake.csv
    True.csv
```

