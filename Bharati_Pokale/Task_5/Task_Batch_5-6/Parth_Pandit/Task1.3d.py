# Task 3(D): Handling punctuation, special characters, emojis

import re
import emoji

text = "Wow!!! This is amazing 😍 #AI #NLP @user123"

print("=== 3(D) TEXT CLEANING ===")
print("Original Text:")
print(text)

clean_text = re.sub(r"#\w+|@\w+", "", text)

clean_text = emoji.replace_emoji(clean_text, replace="")

clean_text = re.sub(r"[^\w\s]", "", clean_text)

clean_text = re.sub(r"\s+", " ", clean_text).strip()

print("\nCleaned Text:")
print(clean_text)
