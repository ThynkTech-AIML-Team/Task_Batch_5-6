from nltk.tokenize import sent_tokenize, word_tokenize

def analyze(paragraph):
    sentences = sent_tokenize(paragraph)
    tokens = word_tokenize(paragraph)

    print("Sentences:", sentences)
    print("Tokens:", tokens)
    print("Sentence count:", len(sentences))
    print("Token count:", len(tokens))
    print("Paragraph count:", len(paragraph.split("\n\n")))

text = "NLP is amazing. It powers many applications."
analyze(text)