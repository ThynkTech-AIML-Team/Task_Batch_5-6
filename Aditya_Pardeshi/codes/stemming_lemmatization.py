import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sample text chapter
text = """
Six months ago ‘Stop, my bhai, stop,’ Saurabh said, snatching away my whisky glass. ‘I am not drunk,’ I said. We were in a corner of the drawing room, near the makeshift bar. The rest of the coaching class faculty had gathered around Arora sir. They would never miss a chance to suck up to him. We had come to the Malviya Nagar house of Chandan Arora, owner of Chandan Classes, and our boss. ‘You swore on me you wouldn’t have more than two drinks,’ Saurabh said. I smiled at him. ‘But did I quantify the size of the drinks? How much whisky per drink? Half a bottle?’ My words slurred. I was finding it hard to balance myself. ‘You need fresh air. Let’s go to the balcony,’ Saurabh said. ‘I need fresh whisky,’ I said. Saurabh dragged me to the balcony by my arm. When had this fatso become so strong? ‘It is freezing here,’ I said, shivering. I rubbed my hands together to keep myself warm. ‘You can’t drink so much, bhai.’ ‘It’s New Year’s Eve. You know what that does to me.’ ‘It’s history. Four years ago. It’s going to be 2018.’ ‘Feels like four seconds ago,’ I said. I took out a cigarette packet, which Saurabh promptly grabbed and hid in his pocket. I pulled out my phone. I opened the contact details of my next intoxicant, Zara. ‘What did she say that night?’ I said, staring at Zara’s WhatsApp profile picture. ‘We are done, that’s what she said. What did she mean done? Howcan she say we? I am not done.’ ‘Leave the phone alone, bhai. You may accidentally call her,’ Saurabh said. He lunged for my phone. I dodged to avoid him. ‘Look at her,’ I said, turning the screen towards Saurabh. She had put up a selfie as her DP—pouting, hand on waist, the black sari a dramatic contrast to her fair, almost pink, face. She didn’t always have her picture as her DP. Often, she would put up quotes. The ‘let life not hold you back’ kinds, statements that sound profound but actually mean nothing. Her WhatsApp display picture was the only connect I had left with her. It was how I knew what was happening in her life. ‘Who wears black saris? She doesn’t look that great,’ Saurabh said. He always did his best to help me get over her. I love Saurabh—my best friend, colleague, and fellow-misfit in this crazy drive called life. He’s from Jaipur, not far from my hometown of Alwar. His father works as a junior engineer in the PWD. Like me, he too didn’t get placed after campus. Both of us worked our asses off at Chandan Classes, even as we hoped to get out of there ASAP. ‘It’s Zara. She always looks great,’ I put it plainly. Saurabh shrugged. ‘That’s part of the tragedy.’ ‘You think I am mad about her because of her looks?’ ‘I think you should shut your phone.’ ‘More than three years, dude. Three crazy, crazy years.’ ‘I know, bhai. If you promise not to drink anymore, we can go back in. It is cold here.’ ‘What do you know?’ ‘That you dated Zara for three years. Want dinner?’ ‘Screw dinner. More than three. Three years, two months and three weeks to be precise.’ ‘You told me. Rendezvous 2010 to New Year’s Eve 2014.’ ‘Yes, Rendezvous. That’s when we met. Did I tell you how we met?’ I said. My feet were finding it harder to find the floor. Saurabh held me tight to prevent me from falling. ‘Yes, you have told me. Fifty times,’ Saurabh muttered. ‘There was a debating competition. She was in the finals.’ ‘Bhai, you have told this story a zillion times,’ he said. I didn’t care. Hecould hear it a zillion times plus one.
"""

# Tokenize text
tokens = word_tokenize(text)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply Stemming
stemmed_words = [stemmer.stem(word) for word in tokens]

# Apply Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Print Results
print("Original Tokens:\n", tokens)
print("\nAfter Stemming:\n", stemmed_words)
print("\nAfter Lemmatization:\n", lemmatized_words)
