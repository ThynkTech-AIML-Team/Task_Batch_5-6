import re
import emoji

def clean_text(text):
    print("Original Text:\n", text)

    # 1️⃣ Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # 2️⃣ Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # 3️⃣ Remove hashtags symbol (keep the word)
    text = re.sub(r'#', '', text)

    # 4️⃣ Remove multiple punctuation (!!!, ???, ...)
    text = re.sub(r'[!?.]{2,}', '', text)

    # 5️⃣ Remove special characters (keep letters and numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 6️⃣ Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    print("\nCleaned Text:\n", text)

    return text


# Example Text
sample_text = "Wow!!! I love this movie 😍😍 #Amazing @john_doe !!! So good??? #NLP 🚀"
sample_text_2="hey ! Good morning 🌞🌻✨"
clean_text(sample_text)
clean_text(sample_text_2)
#optional - converting emoji into text with proper sentiment
print(emoji.demojize(sample_text))