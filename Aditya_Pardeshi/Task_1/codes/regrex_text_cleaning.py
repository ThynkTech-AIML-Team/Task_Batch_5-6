import re

def regex_cleaning(text):
    print("Original Text:\n", text)

    # 1️⃣ Extract all emails
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

    # 2️⃣ Remove numbers
    text_no_numbers = re.sub(r'\d+', '', text)

    # 3️⃣ Replace multiple spaces with single space
    text_cleaned = re.sub(r'\s+', ' ', text_no_numbers).strip()

    print("\nExtracted Emails:")
    print(emails)

    print("\nText After Removing Numbers:")
    print(text_no_numbers)

    print("\nFinal Cleaned Text:")
    print(text_cleaned)

    return emails, text_cleaned


# Example Paragraph
paragraph = """
Contact us at support123@gmail.com or sales.team@company.org.
Call 9876543210 for details. Our office number is 12345.
    Thank    you   for   visiting!
"""

regex_cleaning(paragraph)
