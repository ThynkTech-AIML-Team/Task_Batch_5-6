# Multi-lingual Sentiment Analysis (Important Points)

This notebook builds a simple sentiment analyzer for English and Hindi/other languages.

## What it does
- Detects input language using `langdetect`
- Uses English model for English text: `distilbert-base-uncased-finetuned-sst-2-english`
- Uses multilingual model for non-English text: `nlptown/bert-base-multilingual-uncased-sentiment`
- Converts multilingual star labels into sentiment:
- `1-2 stars` -> Negative
- `3 stars` -> Neutral
- `4-5 stars` -> Positive
- Returns language, model name, sentiment, and confidence score

## Libraries used
- `transformers`
- `langdetect`

## Example coverage
- English sample prediction
- Hindi sample prediction
- Mixed examples for quick validation
