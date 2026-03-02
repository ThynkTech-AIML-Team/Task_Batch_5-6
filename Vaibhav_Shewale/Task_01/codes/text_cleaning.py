import re

def regex_cleaning(text):
    print("Original Text:\n", text)

    # 1. Extract all emails BEFORE removing numbers
    # Added \d to ensure we catch emails with digits in the username
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)

    # 2. Remove numbers 
    # Note: This will affect emails still inside the 'text' variable. 
    # If you want to keep emails intact in the text, you'd need a different approach.
    text_no_numbers = re.sub(r'\d+', '', text)

    # 3. Replace multiple spaces/newlines with a single space
    # \s covers spaces, tabs, and newlines
    text_cleaned = re.sub(r'\s+', ' ', text_no_numbers).strip()

    print("\n--- Results ---")
    print(f"Extracted Emails: {emails}")
    print(f"Final Cleaned Text: {text_cleaned}")

    return emails, text_cleaned

# Example Paragraph
paragraph = """
Contact us at user123@domain.com or sales.team@company.org.
Call 9876543210 for details. Our office number is 12345.
    Thank    you    for    visiting!
"""

emails, cleaned = regex_cleaning(paragraph)