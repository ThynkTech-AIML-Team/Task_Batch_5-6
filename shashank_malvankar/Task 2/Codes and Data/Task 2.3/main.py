from utils.text_loader import load_resume
from utils.ner_extractor import extract_entities
from utils.skill_extractor import extract_skills

if __name__ == "__main__":

    print("Loading resume...")

    text = load_resume(
        "data/sample_resume.txt"
    )

    print("Extracting entities...")

    entities = extract_entities(text)

    print("Extracting skills...")

    skills = extract_skills(text)


    print("\nRESUME PARSER OUTPUT")
    print("----------------------------")

    print("Name:", entities["Name"])

    print("Email:", entities["Email"])

    print("Phone:", entities["Phone"])

    print("Organizations:", entities["Organizations"])

    print("Skills:", skills)
