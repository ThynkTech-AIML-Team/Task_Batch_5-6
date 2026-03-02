# Task 3(F): Regex for Text Cleaning

import re

text = """
Contact us at support@gmail.com or admin123@yahoo.com.
Our office number is 9876543210.
AI 2025 is the future of technology.
"""

print("=== 3(F) REGEX TEXT CLEANING ===")

emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
print("\nExtracted Emails:")
print(emails)

text_no_numbers = re.sub(r"\d+", "", text)
print("\nText After Removing Numbers:")
print(text_no_numbers)

normalized_text = re.sub(r"\s+", " ", text_no_numbers).strip()
print("\nText After Normalizing Spaces:")
print(normalized_text)
