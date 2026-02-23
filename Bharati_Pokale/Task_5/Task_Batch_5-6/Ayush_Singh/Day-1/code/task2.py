import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

def text_statistics(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    tokens = nltk.word_tokenize(paragraph)
    
    # Simple paragraph split by double newline
    paragraph_count = len(paragraph.split('\n\n'))
    
    print("\n--- Task 2: Text Data Basics ---")
    print(f"Sentences: {sentences}")
    print(f"Tokens: {tokens[:10]}...") # Printing first 10 for brevity
    print(f"Token Count: {len(tokens)}")
    print(f"Sentence Count: {len(sentences)}")
    print(f"Paragraph Count: {paragraph_count}")

sample_text = "NLP is fascinating. It powers AI chatbots. \n\nThis is a new paragraph."
text_statistics(sample_text)