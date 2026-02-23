# SMS Spam Detection & Algorithm Comparison

This project implements a binary text classification pipeline to identify **Spam** vs. **Ham** (legitimate) messages. It explores the impact of different text vectorization techniques and compares three popular machine learning algorithms.

## 📊 Dataset
The project uses the **SMS Spam Collection Dataset**, a public set of labeled SMS messages.
* **Total Samples:** 5,572 messages
* **Classes:** Ham (86.6%) and Spam (13.4%)
* **Data Source:** UCI Machine Learning Repository / GitHub Raw TSV

## 🛠️ Project Workflow

### 1. Preprocessing
To clean the raw text and reduce dimensionality, the following steps were applied:
* **Cleaning:** Removal of punctuation and non-alphabetic characters using Regex.
* **Normalization:** Converting all text to lowercase.
* **Stopword Removal:** Filtering out common English words (e.g., "the", "is", "in") using the `nltk` library.

### 2. Feature Engineering
We compared two methods of converting text into numerical vectors:
* **CountVectorizer:** Creates a "Bag of Words" based on simple word frequency.
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their uniqueness, penalizing words that appear too frequently across the entire dataset.



### 3. Models Compared
Three distinct classification algorithms were implemented:
1.  **Multinomial Naive Bayes:** A probabilistic baseline ideal for text data.
2.  **Logistic Regression:** A linear model used to predict the probability of a class.
3.  **Support Vector Machine (SVM):** A high-dimensional classifier using a linear kernel.

---

## 📈 Results & Evaluation

The models were evaluated using **Accuracy**, **Precision/Recall**, and a **Confusion Matrix**.

### Model Performance Comparison
| Vectorizer | Model | Accuracy |
| :--- | :--- | :--- |
| **TF-IDF** | **SVM (Linear)** | **~98.5%** |
| CountVec | Naive Bayes | ~97.8% |
| TF-IDF | Logistic Regression | ~96.2% |



### Key Insights
* **Spam Indicators:** Words like *free, txt, claim, stop, mobile,* and *reply* showed the highest feature importance for the Spam class.
* **Vectorizer Impact:** TF-IDF generally reduced False Positives (legitimate mail marked as spam) compared to simple counting.
* **Top Performer:** The **Linear SVM with TF-IDF** achieved the highest overall accuracy.

---

