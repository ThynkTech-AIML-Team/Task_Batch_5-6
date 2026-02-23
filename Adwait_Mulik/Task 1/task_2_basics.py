import nltk
import os

# Professional Standard: Ensure the tokenizer models are available
nltk.download('punkt')

def analyze_text_structure(paragraph):
    """
     Function to extract structural NLP components from a text block.
    """
    # 1. Generate List of Sentences [cite: 11]
    # nltk.sent_tokenize handles punctuation better than standard .split()
    sentences = nltk.sent_tokenize(paragraph)
    
    # 2. Generate List of Tokens (words and punctuation) [cite: 12]
    tokens = nltk.word_tokenize(paragraph)
    
    # 3. Identify Paragraphs [cite: 13]
    # In professional text processing, paragraphs are separated by newlines
    paragraph_list = [p for p in paragraph.split('\n') if p.strip()]
    
    # Return structured data [cite: 13]
    return {
        "sentence_list": sentences,
        "token_list": tokens,
        "metrics": {
            "Total Tokens": len(tokens),
            "Total Sentences": len(sentences),
            "Total Paragraphs": len(paragraph_list)
        }
    }

if __name__ == "__main__":
    print("--- STARTING TASK 2: TEXT DATA BASICS ---")
    
    # Sample multi-paragraph input for testing
    sample_text = """Natural Language Processing is a multidisciplinary field. It is used at ThynkTech.
    
    This is paragraph number two. It focuses on the structural basics of text data."""
    
    results = analyze_text_structure(sample_text)
    
    # Display results to the terminal
    print(f"Metrics Found: {results['metrics']}")
    print(f"Sentences: {results['sentence_list']}")
    
    # Save the structural report to your 'outputs' folder for the office repo
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/task_2_report.txt', 'w') as f:
        f.write("OFFICE NLP REPORT: TEXT STRUCTURE\n")
        f.write("=================================\n")
        for key, value in results['metrics'].items():
            f.write(f"{key}: {value}\n")
            
    print("\n[SUCCESS] Task 2 complete. Report saved in 'outputs/task_2_report.txt'.")