import re
import emoji

def clean_text(text):
    # 1. Remove URLs FIRST
    text = re.sub(r'http\S+|www\S+', '', text)

    # 2. Convert Emojis to text (Optional but recommended for sentiment)

    text = emoji.demojize(text)

    # 3. Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # 4. Remove Hashtag symbols but keep the word
    text = text.replace('#', '')

    # 5. Remove special characters 

    text = re.sub(r'[^a-zA-Z0-9\s_!?. ]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower() 


samples = [
    "BUYER BEWARE!!! 🛑 The product broke in 2 days... @CompanyHelp fix this! http://bit.ly/refund #fail #scam",
    "Check out my new blog post 📝: https://example.com/nlp-tips #learning #coding_is_fun"
]

for s in samples:
    print(f"CLEANED: {clean_text(s)}")