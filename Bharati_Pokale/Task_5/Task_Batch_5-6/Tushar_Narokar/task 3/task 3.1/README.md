# Text Classification Project

## Dataset
The project utilizes the SMS Spam Collection dataset. To ensure offline functionality and avoid network issues, a representative sample of 7,700 messages is embedded directly within the notebook.

## Models Used
Three classification algorithms were implemented and compared:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (LinearSVC)

Each model was tested using two different text vectorization techniques:
1. CountVectorizer (Bag of Words)
2. TfidfVectorizer (Term Frequency-Inverse Document Frequency)

## Results
All models demonstrated high performance on the dataset. Logistic Regression and SVM showed slightly better precision in identifying spam compared to Naive Bayes. TF-IDF vectorization generally provided a minor accuracy boost over CountVectorizer. Detailed metrics including Confusion Matrices and Classification Reports are provided in the notebook.
