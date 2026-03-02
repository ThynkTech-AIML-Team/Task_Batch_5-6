import spacy  # type: ignore
import os

# Professional check: Load the NLP model or download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("📥 Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    print("🧠 Analyzing resume text with NER...")
    doc = nlp(text)
    
    # Task: Extract skills, education, organizations, names
    results = {
        "Names": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "Organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        "Locations": [ent.text for ent in doc.ents if ent.label_ == "GPE"],
        "Education/Other": [ent.text for ent in doc.ents if ent.label_ in ["FAC", "DEGREE"]]
    }
    return results

if __name__ == "__main__":
    # Example input text
    sample_resume = """
    Adwait Mulik is a Computer Engineering student at ISBM College of Engineering in Pune. 
    He is highly skilled in Python, SQL, and Machine Learning.
    """
    
    entities = extract_entities(sample_resume)
    
    # Save Final Report
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/resume_parser_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== NER RESUME PARSER RESULTS ===\n\n")
        for label, values in entities.items():
            f.write(f"{label}: {', '.join(values)}\n")
            
    print(f"✅ Final Task Complete! Report saved at: {report_path}")