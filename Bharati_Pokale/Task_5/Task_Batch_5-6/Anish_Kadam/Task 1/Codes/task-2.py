import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

#Download Tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

def text_data_basics(paragraph):
    
    
    sentences = sent_tokenize(paragraph) #tokenize sentences
    
    tokens = word_tokenize(paragraph) #tokenize words
    
    paragraphs = [p for p in paragraph.split("\n") if p.strip() != ""] #paragraph count
    
    # Print results
    print("List of Sentences :")
    for s in sentences:
        print("-->", s)
    
    print("\nList of Tokens :")
    print(tokens)
    
    print("Number of Sentences:", len(sentences))
    print("Number of Tokens:", len(tokens))
    print("Number of Paragraphs:", len(paragraphs))


#User Input
print("Enter your paragraph (Press ENTER twice to finish):")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

text = "\n".join(lines)

text_data_basics(text)

