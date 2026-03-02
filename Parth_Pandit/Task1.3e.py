# Task 3(E): Lowercasing & Normalization

import re

text = "  Natural   Language PROCESSING   Is   VERY   Powerful   "

print("=== 3(E) LOWERCASE & NORMALIZATION ===")
print("Original Text:")
print(f"'{text}'")

text = text.lower()

text = re.sub(r"\s+", " ", text).strip()

print("\nProcessed Text:")
print(f"'{text}'")
