import pandas as pd
import nltk
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Professional standard: Download required NLP resources [cite: 5]
nltk.download('punkt')

def run_task_1():
    try:
        # Load the 15-row open dataset [cite: 5]
        df = pd.read_csv('data/movie_reviews.csv')
        
        # --- 1. Sentiment Analysis [cite: 6] ---
        def get_sentiment(text):
            blob = TextBlob(str(text))
            # Use getattr to bypass Pylance attribute/indexing errors
            # This is a robust way to get 'polarity' safely
            sentiment_obj = blob.sentiment
            polarity = getattr(sentiment_obj, 'polarity', 0)
            
            return "Positive" if polarity > 0 else "Negative"

        # Apply labeling to the first 13 rows [cite: 5]
        df['sentiment'] = df.iloc[:13]['content'].apply(get_sentiment)
        
        print("--- TASK 1: SENTIMENT ANALYSIS RESULTS ---")
        print(df[['review_id', 'sentiment']].head(13))

        # --- 2. Text Summarization [cite: 7] ---
        print("\n--- TASK 1: TEXT SUMMARIZATION ---")
        if len(df) >= 14:
            # Summarizing the technical text at index 13
            long_text = str(df.iloc[13]['content'])
            parser = PlaintextParser.from_string(long_text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 1)

            if len(summary) > 0:
                print(f"Summary: {summary[0]}")
        
        # Save output for office repo tracking
        df.to_csv('outputs/task_1_final_results.csv', index=False)
        print("\n[SUCCESS] Task 1 results saved in 'outputs' folder.")

    except Exception as e:
        print(f"Error executing Task 1: {e}")

if __name__ == "__main__":
    run_task_1()