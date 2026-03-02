from nltk.tokenize import sent_tokenize, word_tokenize

def analyze_text(paragraph):

    sentences = sent_tokenize(paragraph)
    
 
    tokens = word_tokenize(paragraph)

    paragraphs = [p for p in paragraph.split('\n') if p.strip() != ""]
    
    # Counts
    sentence_count = len(sentences)
    token_count = len(tokens)
    paragraph_count = len(paragraphs)
    
    return {
        "Sentences": sentences,
        "Tokens": tokens,
        "Sentence Count": sentence_count,
        "Token Count": token_count,
        "Paragraph Count": paragraph_count
    }


text = '''The city wakes up before the sun does. Streetlights flicker like tired guardians while milk vans hum through narrow lanes. A tea seller arranges small glasses in perfect symmetry, waiting for the first customer who will sip warmth before the world becomes loud. Somewhere above, a lone pigeon .

By noon, the rhythm changes. Offices buzz with quiet ambition, keyboards tapping like mechanical rain. In a nearby park, an old man feeds crumbs to sparrows while pretending not to notice how time slips through his fingers. Children race bicycles in uneven circles.

When evening arrives, the city exhales. Windows glow like scattered constellations, each holding its own private story. Conversations float across balconies, blending with distant traffic and the aroma of dinner spices. '''

result = analyze_text(text)

for key, value in result.items():
    print(f"{key}:\n{value}\n")