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

sample_text = """India is widely recognized as the "Pharmacy of the World," a title earned through its massive production capacity and role as a global supplier of affordable generic medicines. Currently, the nation ranks 3rd globally in pharmaceutical production by volume and 14th by value, accounting for approximately 20% of the global supply of generic drugs.

The industry's success is rooted in the Patents Act of 1970, which allowed Indian companies to reverse-engineer branded drugs to develop cost-effective alternatives. Today, India is the largest provider of generic medicines globally, meeting 40% of the generic demand in the United States and supplying roughly 60% of the world's vaccines. Major hubs like Hyderabad and Mumbai house numerous USFDA-compliant manufacturing facilities, the highest number outside the U.S..

Looking ahead to 2026, the sector is transitioning from a cost-driven model to one focused on high-value innovation, including biosimilars and complex therapeutics. Technologies like Artificial Intelligence (AI) and Natural Language Processing (NLP) are being integrated into R&D and clinical trials to accelerate drug discovery and optimize supply chains. Despite challenges like regulatory hurdles and raw material dependence, India remains a critical partner in global healthcare security."""
text_statistics(sample_text)