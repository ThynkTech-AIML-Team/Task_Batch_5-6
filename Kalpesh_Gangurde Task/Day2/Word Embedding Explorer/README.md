# Word Embedding Explorer (Important Points)

This notebook compares word embedding models and shows how they behave on common NLP tasks.

## What it covers
- Loads embedding models using `gensim`
- Uses small demo setup for fast run (`load_full = False`)
- Compares FastText, GloVe, and toy Word2Vec/FastText models
- Similarity checks (example: `king` vs `queen`)
- Analogy task (`king - man + woman`)
- OOV (Out-Of-Vocabulary) behavior comparison
- Word-space visualization using PCA and optional t-SNE

## Key takeaway
- FastText can handle many unseen words better because of subword modeling.

## Libraries used
- `gensim`
- `numpy`
- `matplotlib`
- `scikit-learn`
