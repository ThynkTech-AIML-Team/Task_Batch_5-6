import nltk
nltk.download('punkt_tab')

def text_info(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    tokens = nltk.word_tokenize(paragraph)

    print("Sentences:", sentences)
    print("Tokens:", tokens)
    print("Sentence Count:", len(sentences))
    print("Token Count:", len(tokens))
    print("Paragraph Count:", paragraph.count('\n') + 1)

text = "NLP is interesting. It helps machines understand language."

text_info(text)
