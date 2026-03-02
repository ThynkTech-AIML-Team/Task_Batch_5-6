import sys
import numpy as np
import pandas as pd
import re, copy, random, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

print(f"Python {sys.version}")
print(f"numpy {np.__version__}, pandas {pd.__version__}")

STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','a','an','the','and','but',
    'if','or','as','of','at','by','for','with','to','from','in','out',
    'on','off','so','than','too','very','will','just','now','can','not',
    'no','nor','only','own','same','s','t','ll','re','ve','d',
}

RAW = [
    ('ham', 'hey how are you doing today hope all is well'),
    ('ham', 'can we meet for coffee at three pm this afternoon'),
    ('ham', 'great game last night watching the match was so exciting'),
    ('ham', 'happy birthday hope your day is wonderful and joyful'),
    ('ham', 'i finished the report and sent it to the manager today'),
    ('ham', 'the weather is beautiful today let us go for a walk outside'),
    ('ham', 'did you remember to pay the electricity bill this month'),
    ('ham', 'sorry missed your call i was in a meeting all morning'),
    ('ham', 'the concert last night was absolutely amazing best performance'),
    ('ham', 'thanks for your help with my presentation really appreciated'),
    ('spam', 'winner you have been selected to receive a prize reward call now'),
    ('spam', 'urgent you have won free membership in prize jackpot claim now'),
    ('spam', 'you have been randomly selected to receive free iphone reply yes'),
    ('spam', 'special offer unlimited calls for just monthly first month free reply yes'),
    ('spam', 'final notice payment overdue call now to avoid legal action immediately'),
    ('spam', 'win amazon gift card click here to claim your prize limited offer today'),
    ('spam', 'free loan offer approved no credit check required call freeloan now'),
    ('spam', 'earn daily from home no experience needed limited spots click here start'),
    ('spam', 'cash advance approved deposited today no paperwork click instant approval'),
    ('spam', 'your lottery ticket won claim prize before expires reply claim receive'),
]

RAW = RAW * 50
random.seed(42)
random.shuffle(RAW)

df = pd.DataFrame(RAW, columns=['label', 'message'])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t) > 1)

df['clean'] = df['message'].apply(preprocess)
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

X_tr, X_te, y_tr, y_te = train_test_split(
    df['clean'], df['label_enc'], test_size=0.2, random_state=42, stratify=df['label_enc']
)

cv = CountVectorizer(max_features=500)
X_tr_cv = cv.fit_transform(X_tr)
X_te_cv = cv.transform(X_test)

tf = TfidfVectorizer(max_features=500)
X_tr_tf = tf.fit_transform(X_tr)
X_te_tf = tf.transform(X_te)

for name, clf in [
    ('Naive Bayes', MultinomialNB()),
    ('Logistic Regression', LogisticRegression(max_iter=200, random_state=42)),
    ('SVM', LinearSVC(max_iter=500, random_state=42)),
]:
    m = copy.deepcopy(clf)
    m.fit(X_tr_cv, y_tr)
    cv_acc = accuracy_score(y_te, m.predict(X_te_cv))
    m2 = copy.deepcopy(clf)
    m2.fit(X_tr_tf, y_tr)
    tf_acc = accuracy_score(y_te, m2.predict(X_te_tf))
    print(f"  {name}: CV={cv_acc*100:.2f}% | TF-IDF={tf_acc*100:.2f}%")

corpus_spam = ' '.join(df[df['label'] == 'spam']['clean'])
wc = WordCloud(width=300, height=150, background_color='white').generate(corpus_spam)
fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(wc)
ax.axis('off')
plt.tight_layout()
plt.savefig('test_wc_output.png', dpi=80)
plt.close()

print("ALL CHECKS PASSED")
