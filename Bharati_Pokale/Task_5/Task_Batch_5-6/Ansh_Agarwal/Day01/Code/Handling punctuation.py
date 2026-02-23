import re
import emoji

text = "Hello @user!!! NLP is awesome 😊 #AI #MachineLearning"

text_no_emoji = emoji.replace_emoji(text, replace='')

cleaned_text = re.sub(r'[@#]\w+', '', text_no_emoji)  
cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)   

print("Original:", text)
print("Cleaned:", cleaned_text)