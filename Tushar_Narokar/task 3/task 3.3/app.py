import json
import nltk
import heapq
import string
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def nltk_setup():
    """Ensure all required NLTK data is available."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"NLTK setup failed: {e}")

class TextSummarizer:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or set(stopwords.words('english'))

    def frequency_based(self, text, limit=None):
        """Frequency-based summarization."""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1: return text
        
        words = word_tokenize(text.lower())
        freq_table = {}
        for word in words:
            if word not in self.stop_words and word not in string.punctuation:
                freq_table[word] = freq_table.get(word, 0) + 1
        
        if not freq_table: return text
        
        max_freq = max(freq_table.values())
        for word in freq_table: freq_table[word] /= max_freq
        
        sent_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in freq_table:
                    sent_scores[sent] = sent_scores.get(sent, 0) + freq_table[word]
        
        n = limit or max(1, int(len(sentences) * 0.4))
        summary_sentences = heapq.nlargest(n, sent_scores, key=sent_scores.get)
        return " ".join([s for s in sentences if s in summary_sentences])

    def text_rank(self, text, limit=None):
        """TextRank summarization using sentence similarity graph."""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1: return text
        
        # Vectorize sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Rank sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        n = limit or max(1, int(len(sentences) * 0.4))
        top_sentences = [ranked_sentences[i][1] for i in range(min(n, len(ranked_sentences)))]
        
        # Return in original order
        return " ".join([s for s in sentences if s in top_sentences])

def display_results(original, summarized, method_name, stats_list):
    print(f"\n{'='*20} {method_name} {'='*20}")
    print(f"Summary: {summarized}")
    
    orig_words = len(original.split())
    summ_words = len(summarized.split())
    reduction = (1 - (summ_words / orig_words)) * 100
    
    stats_list.append({
        "Method": method_name,
        "Original Words": orig_words,
        "Summary Words": summ_words,
        "Reduction %": f"{reduction:.1f}%"
    })

if __name__ == "__main__":
    nltk_setup()
    summarizer = TextSummarizer()
    all_stats = []

    print("\n" + "="*50)
    print("      AUTOMATIC TEXT SUMMARIZER")
    print("="*50)
    
    print("\nEnter the text you want to summarize.")
    print("If you leave it blank and press Enter, the application will use 'dataset.json'.")
    user_text = input("\nYour Text: ").strip()

    if user_text:
        print(f"\n[INFO] Processing your input...")
        # Frequency Based
        freq_summary = summarizer.frequency_based(user_text)
        display_results(user_text, freq_summary, "Frequency-Based", all_stats)
        
        # TextRank
        tr_summary = summarizer.text_rank(user_text)
        display_results(user_text, tr_summary, "TextRank", all_stats)
    else:
        print("\n[INFO] No input detected. Loading 'dataset.json'...")
        try:
            with open('dataset.json', 'r') as f:
                data = json.load(f)
            
            for article in data:
                print(f"\n\n--- Processing Article: {article['title']} ---")
                print(f"Original Content: {article['text'][:200]}...") # Print preview
                
                # Frequency Based
                freq_summary = summarizer.frequency_based(article['text'])
                display_results(article['text'], freq_summary, "Frequency-Based", all_stats)
                
                # TextRank
                tr_summary = summarizer.text_rank(article['text'])
                display_results(article['text'], tr_summary, "TextRank", all_stats)
        except FileNotFoundError:
            print("[ERROR] dataset.json not found in the current directory.")

    if all_stats:
        print("\n\n" + "="*50)
        print("GLOBAL PERFORMANCE COMPARISON")
        print("="*50)
        df = pd.DataFrame(all_stats)
        print(df.to_string(index=False))
        print("="*50)

