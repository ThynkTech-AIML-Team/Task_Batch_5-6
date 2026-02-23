# Topic Modeling Comparison

Analysis of LDA, NMF, and BERTopic on 20 Newsgroups dataset.

## Files

- topic_modeling.ipynb - Main analysis notebook
- lda_topics.png - LDA topic visualization
- nmf_topics.png - NMF topic visualization

## How to Run

Open the notebook:
```
jupyter notebook topic_modeling.ipynb
```

## What's Inside

- LDA: 10 topics, Log Likelihood: -5,422,287
- NMF: 10 topics, Reconstruction Error: 0.000184
- BERTopic: Transformer-based topic extraction
- Visualizations: PNG files + inline plots
- Metrics: Coherence scores, model comparison

## Quick Results

| Model | Quality | Speed |
|-------|---------|-------|
| LDA | Good | Very Fast |
| NMF | Excellent | Very Fast |
| BERTopic | Best | Slow |

Done!
