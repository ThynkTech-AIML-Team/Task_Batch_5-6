Dataset -: spam.csv 
Download link -: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data
-------------------------------------------------------------------------------------------------------------------------------------------------------
Models used -: 1. Naive Bayes (Multinomial)
               2. Logistic Regression
               3. Support Vector Machine
-------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries Used -: numpy, pandas, matplotlib, seaborn, string, nltk, scikit-learn.
-------------------------------------------------------------------------------------------------------------------------------------------------------
### Results -:

**Table 1 : Accuracy**
| Algorithms | CountVectorized | TF-IDF |
| -------- | -------- | -------- |
| Naive Bayes | 0.9757847533632287 | 0.9659192825112107 |
| Logistic Regression | 0.9775784753363229 | 0.9426008968609866 |
| Support Vector Machine | 0.9730941704035875 | 0.967713004484305 |

Top Spam Words: ['service' 'text' 'prize' 'reply' 'call' 'mobile' 'free' 'stop' 'claim' 'txt']

Top Ham Words: ['ltgt' 'im' 'ill' 'ok' 'sir' 'come' 'later' 'da' 'going' 'got']

**Note-: my test size and random state values are 0.2 & 42 respectively.**
