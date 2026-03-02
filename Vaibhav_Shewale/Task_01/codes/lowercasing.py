def normalize_text(text):
    #converting to lowercase
    text = text.lower()
    
    normalized = " ".join(text.split())
    
    return normalized

# Test Samples
samples = [
    "  Mixed CASE   text WITH   too MANY spaces  ",
    "Line breaks\nand\ttabs\rshould be replaced",
    "ALREADY LOWERED BUT    GAPPY"
]

for s in samples:
    print(f"Result: '{normalize_text(s)}'")