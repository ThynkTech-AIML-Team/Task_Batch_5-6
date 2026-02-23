import spacy
import re

nlp = spacy.load("en_core_web_sm")

def load_skills(file_path):
    with open(file_path, "r") as f:
        return [skill.strip().lower() for skill in f.readlines()]

skills_db = load_skills("skills_list.txt")

def extract_skills(text):
    text = text.lower()
    found_skills = set()
    for skill in skills_db:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.add(skill)
    return list(found_skills)

def parse_resume(text):
    doc = nlp(text)

    names = []
    education = []
    organizations = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)
        elif ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ in ["GPE", "NORP"]:
            education.append(ent.text)

    skills = extract_skills(text)

    return {
        "Name": list(set(names)),
        "Education": list(set(education)),
        "Organizations": list(set(organizations)),
        "Skills": skills
    }

if __name__ == "__main__":
    with open("sample_resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    parsed_data = parse_resume(resume_text)

    for key, value in parsed_data.items():
        print(f"{key}: {value}")
