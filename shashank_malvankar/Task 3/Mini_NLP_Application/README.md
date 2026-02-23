# Automatic Text Summarizer - Mini NLP Application

## Overview

This project implements an automatic text summarization system using Natural Language Processing (NLP). The summarizer reduces long text into a shorter version while preserving the most important information.

A frequency-based extractive summarization approach was used to identify and extract the most relevant sentences.

---

## Objective

The objective of this task was to:

- Implement an automatic text summarization system
- Use frequency-based summarization
- Display original text and summarized text
- Compare original and summary lengths
- Demonstrate basic NLP preprocessing techniques

---

## Method Used: Frequency-Based Extractive Summarization

This approach works by:

1. Tokenizing the text into words and sentences
2. Removing stopwords and punctuation
3. Calculating word frequencies
4. Assigning scores to sentences based on word importance
5. Selecting the highest scoring sentences to form the summary

This method extracts important sentences directly from the original text.

---

## NLP Techniques Used

- Text tokenization
- Stopword removal
- Word frequency analysis
- Sentence scoring
- Extractive summarization

Libraries used:

- NLTK (Natural Language Toolkit)
- Python standard libraries

---

## Example

### Original Text Length:
Example: 85 words

### Summary Length:
Example: 32 words

The summarizer successfully reduced the text size while preserving key information.

---

## Results and Observations

- Important sentences were correctly identified
- Summary retained core meaning of the original text
- Significant reduction in text length was achieved
- Frequency-based approach proved effective for extractive summarization

---

## Files Included

- `text_summarizer.ipynb` : Jupyter Notebook containing implementation
- `outputs/` : Screenshots of original text and summary
- `README.md` : Project documentation

---

## Conclusion

This project demonstrates a basic automatic text summarization system using NLP techniques. Frequency-based summarization effectively extracts important sentences and reduces text length while preserving essential meaning.

Text summarization is widely used in real-world applications such as:

- News summarization
- Document summarization
- Search engines
- Chatbots
- Content recommendation systems
