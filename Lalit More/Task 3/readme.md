1. Text Classification - spam detection
    Dataset Used: SMS Spam Collection   Dataset
        -> Contains labeled sms messages as spam or ham
        -> binary classification problem
    Preprocessing steps:
        -> convert text to lowercase
        -> remove punctuation
        -> remove stopwords
        -> train test split
    Model used:
        -> Logistic regression
        -> Naive bayes
        -> Support Vector Machine
    Results:
        -> Logistic regression achieved highest accuracy among others approx 0.977
        -> Naive Bayes performed well but slightly lower than Logistic Regression

2. Word Embedding 
    Model used:
        -> Word2Vec, Glove
           Both models were loaded using Gensim library
    Similarity search:
        -> i tested similarity search for word "king"
    Word2Vec vs GloVe:
        -> Both models produced similar related words.Word2Vec showed slightly stronger results in analogy tasks

3. Mini NLP application- Fake news Detection
    Dataset used: Fake and real news dataset
    -> Dataset contains news articles labeled as: Fake, Real
    Model used:
     -> Logistic regression
     -> Naive bayes
    Results: 
    -> Logistic Regression achieved higher accuracy approx 98%
    -> Naive Bayes performed slightly lower approx 94%
