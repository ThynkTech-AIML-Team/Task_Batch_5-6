# Mini NLP Application: Automatic Text Summarizer

This application implements two different methods for extracting summaries from long-form text articles. It is built using Python 3.11 and leverages the NLTK and NetworkX libraries.

## 📂 Dataset
The application uses a custom JSON-based dataset (`dataset.json`) consisting of short articles on various topics:
- **SpaceX Starship Progress**: Aerospace technology and interplanetary goals.
- **Quantum Computing Breakthrough**: Computational physics and engineering.
- **Renewable Energy Growth**: Environmental science and global energy trends.

## 🤖 Models & Methods Used

### 1. Frequency-Based Summarizer
- **Logic**: This method calculates the frequency of each word (excluding stop words and punctuation). Sentences are then scored based on the sum of the importance of the words they contains.
- **Optimization**: Word frequencies are normalized by the maximum frequency to ensure fair scoring across different lengths of text.
- **Type**: Extractive Summarization.

### 2. TextRank Summarizer
- **Logic**: A graph-based ranking algorithm inspired by Google's PageRank. It treats each sentence as a node in a graph.
- **Similarity**: Edges between sentences are weighted using **Cosine Similarity** of their TF-IDF representations.
- **Algorithm**: The PageRank algorithm is applied to the graph to find the most "central" or significant sentences.
- **Type**: Graph-based Extractive Summarization.

## 📊 Results & Performance
The application compares the original text length against the summarized versions and provides statistics on word count reduction.

### Typical Performance Stats:
| Method | Original Words | Summary Words | Reduction % |
|--------|----------------|---------------|-------------|
| Frequency-Based | ~100 | ~35 | ~65% |
| TextRank | ~100 | ~30 | ~70% |

### Key Observations:
- **TextRank** tends to produce more cohesive summaries by identifying sentences that share the most common themes with the rest of the text.
- **Frequency-Based** is faster and works well for texts with clear repeating keywords.
- Both methods successfully reduced the corpus by over **60%** while maintaining the core context.

## 🚀 How to Run
1. Ensure Python 3.11 is installed.
2. The virtual environment is already set up in `venv/`.
3. Run the application:
   ```bash
   .\venv\Scripts\python.exe app.py
   ```
4. **Interactive Mode**:
   - The app will prompt you for text input.
   - **Custom Input**: Paste any long paragraph to get a summary.
   - **Default Mode**: Simply press **Enter** without typing anything to process the pre-loaded `dataset.json`.
