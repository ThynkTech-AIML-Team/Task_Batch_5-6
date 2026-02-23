import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = """
The productivity of the programmers was increasing significantly as they were 
studying the complexities of the underlying algorithms. Several revolutionary 
technologies are currently being developed by various companies. Scientists 
have been studying these phenomena for decades, documenting how different 
species are adapting to changing environments. My friend's feet were aching 
after he had been walking through the mossy forests, but he felt that the 
stunning views were worth the struggle.
"""

tokens = word_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in tokens]

lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Print Results
print("Original Tokens:\n", tokens)
print("\nAfter Stemming:\n", stemmed_words)
print("\nAfter Lemmatization:\n", lemmatized_words)