import re

text = "Contact us at info@test.com or support123@gmail.com. Call 9876543210 now."

emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)

no_numbers = re.sub(r'\d+', '', text)

normalized = re.sub(r'\s+', ' ', no_numbers)

print("Emails:", emails)
print("Without Numbers:", no_numbers)
print("Normalized:", normalized)