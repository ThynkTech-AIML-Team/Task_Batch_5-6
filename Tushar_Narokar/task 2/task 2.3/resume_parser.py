import re, sys, os, json
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")




SKILLS_DB = {
    "python", "java", "javascript", "c++", "c#", "r", "go", "rust", "swift",
    "kotlin", "typescript", "php", "ruby", "scala", "matlab",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "fastapi", "spring", "express", "rest api", "graphql",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "hugging face", "transformers",
    "sql", "mysql", "postgresql", "mongodb", "redis", "sqlite",
    "elasticsearch", "firebase",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "ci/cd", "linux", "terraform", "jenkins",
    "excel", "tableau", "power bi", "spark", "hadoop", "kafka",
    "agile", "scrum", "jira",
}

EDUCATION_KEYWORDS = {
    "bachelor", "master", "phd", "b.tech", "m.tech", "b.sc", "m.sc",
    "b.e", "m.e", "mba", "bca", "mca", "diploma", "degree", "university",
    "college", "institute", "school", "engineering", "science", "arts",
    "technology", "graduate", "undergraduate", "b.s.", "m.s.",
}

EXPERIENCE_KEYWORDS = {
    "engineer", "developer", "analyst", "manager", "intern", "lead",
    "architect", "consultant", "designer", "scientist", "researcher",
    "executive", "coordinator", "specialist", "associate", "officer",
}


import pytesseract
_TESS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for _p in _TESS_PATHS:
    if os.path.exists(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        break


def _pdf_extract_pymupdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _pdf_extract_ocr(path: str) -> str:
    import fitz
    from PIL import Image
    import io
    doc = fitz.open(path)
    texts = []
    for page in doc:
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        texts.append(pytesseract.image_to_string(img, lang="eng"))
    return "\n".join(texts)


def extract_text(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text = _pdf_extract_pymupdf(path)
        if text.strip():
            return text
        print("    [PDF has no selectable text — running OCR...]")
        return _pdf_extract_ocr(path)

    elif ext == "docx":
        import docx2txt
        return docx2txt.process(path)

    else:
        with open(path, encoding="utf-8") as f:
            return f.read()


def extract_email(text):
    match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match[0] if match else None

def extract_phone(text):
    match = re.findall(r"(\+?\d[\d\s\-().]{8,14}\d)", text)
    return match[0].strip() if match else None

_NOT_A_NAME = SKILLS_DB | {
    "tensorflow", "pytorch", "pandas", "numpy", "matplotlib", "keras",
    "scikit", "docker", "kubernetes", "github", "linkedin",
}

def extract_name(doc, email, phone):
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if "\n" in name:
                continue
            if email and email in name:
                continue
            if name.lower() in _NOT_A_NAME:
                continue
            words = name.split()
            if 2 <= len(words) <= 5:
                return name
    for line in doc.text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if email and email in line:
            continue
        if phone and phone.replace(" ", "") in line.replace(" ", ""):
            continue
        if re.search(r"http|www|linkedin|@|EDUCATION|EXPERIENCE|SKILLS|ORGANIZATIONS", line, re.IGNORECASE):
            continue
        words = line.split()
        if (2 <= len(words) <= 4
                and all(w[0].isupper() for w in words if w.isalpha())
                and not any(char.isdigit() for char in line)
                and line.lower() not in _NOT_A_NAME):
            return line
    return None

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in sorted(SKILLS_DB):
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found

def extract_education(text):
    lines = text.split("\n")
    education, seen = [], set()
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in EDUCATION_KEYWORDS):
            cleaned = line.strip()
            if cleaned and len(cleaned) > 5 and cleaned not in seen:
                seen.add(cleaned)
                education.append(cleaned)
    return education[:6]

def extract_organizations(doc):
    seen, orgs = set(), []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            o = " ".join(ent.text.split())
            if not o or len(o) <= 1:
                continue
            if o.lower() in _NOT_A_NAME:
                continue
            if re.search(r"[|@\-]{2,}|^\d+$", o):
                continue
            key = o.lower()
            if key not in seen:
                seen.add(key)
                orgs.append(o)
    return orgs

def extract_experience(text):
    lines = text.split("\n")
    experience, seen = [], set()
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in EXPERIENCE_KEYWORDS):
            cleaned = line.strip()
            if cleaned and len(cleaned) > 8 and cleaned not in seen:
                seen.add(cleaned)
                experience.append(cleaned)
    return experience[:8]




def parse_resume(text: str, filename: str = "N/A") -> dict:
    doc = nlp(text)
    email = extract_email(text)
    phone = extract_phone(text)
    return {
        "filename":      filename,
        "name":          extract_name(doc, email, phone),
        "email":         email,
        "phone":         phone,
        "skills":        extract_skills(text),
        "education":     extract_education(text),
        "organizations": extract_organizations(doc),
        "experience":    extract_experience(text),
    }



def print_results(result: dict):
    bar = "─" * 60
    print(f"\n{bar}")
    print("  RESUME PARSER — EXTRACTED ENTITIES")
    print(f"{bar}\n")

    def show(label, value):
        if not value:
            print(f"  {label:<18} —")
        elif isinstance(value, list):
            if value:
                print(f"  {label:<18} {value[0]}")
                for item in value[1:]:
                    print(f"  {'': <18} {item}")
            else:
                print(f"  {label:<18} —")
        else:
            print(f"  {label:<18} {value}")
        print()

    show("File",          result["filename"])
    show("Name",          result["name"])
    show("Email",         result["email"])
    show("Phone",         result["phone"])
    show("Skills",        result["skills"])
    show("Education",     result["education"])
    show("Organizations", result["organizations"])
    show("Experience",    result["experience"])
    print(bar)




def save_json(results: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"  → JSON : {path}")

def save_csv(results: list, path: str):
    rows = []
    for r in results:
        rows.append({
            "filename":      r["filename"],
            "name":          r["name"] or "",
            "email":         r["email"] or "",
            "phone":         r["phone"] or "",
            "skills":        ", ".join(r["skills"]),
            "education":     " | ".join(r["education"]),
            "organizations": ", ".join(r["organizations"]),
            "experience":    " | ".join(r["experience"]),
        })
    df = pd.DataFrame(rows, columns=[
        "filename", "name", "email", "phone",
        "skills", "education", "organizations", "experience"
    ])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → CSV  : {path}")




SAMPLE_RESUME = """
Priya Sharma
priya.sharma@email.com | +91-9876543210 | LinkedIn: linkedin.com/in/priyasharma

EDUCATION
B.Tech in Computer Science — IIT Bombay, 2021
M.Tech in Artificial Intelligence — IIT Delhi, 2023

EXPERIENCE
Machine Learning Engineer — Google India, 2023–Present
  - Built NLP pipelines using PyTorch and Hugging Face Transformers
  - Deployed models on AWS using Docker and Kubernetes

Data Science Intern — Flipkart, 2022
  - Performed sentiment analysis using scikit-learn and pandas
  - Created dashboards in Tableau and Power BI

SKILLS
Python, TensorFlow, PyTorch, Scikit-learn, NLP, Deep Learning,
SQL, MongoDB, Docker, AWS, Git, REST API, Flask, React

ORGANIZATIONS
Google India, Flipkart, IIT Bombay, IIT Delhi, ACM Student Chapter
"""



if __name__ == "__main__":
    results = []

    if len(sys.argv) > 1:
        path = sys.argv[1]

        if os.path.isdir(path):
            print(f"\nProcessing directory: {path}")
            files = [f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))
                     and f.lower().endswith((".pdf", ".docx", ".txt"))]
            print(f"Found {len(files)} resume(s): {', '.join(files)}\n")
            for filename in files:
                file_path = os.path.join(path, filename)
                print(f"  Analyzing: {filename} ...", end=" ")
                try:
                    text = extract_text(file_path)
                    res = parse_resume(text, filename)
                    results.append(res)
                    print("✓")
                except Exception as e:
                    print(f"✗ ({e})")
        else:
            print(f"\nReading file: {path}")
            try:
                text = extract_text(path)
                res = parse_resume(text, os.path.basename(path))
                results.append(res)
                print_results(res)
            except Exception as e:
                print(f"Error: {e}")

    else:
        print("\nNo file provided — running on built-in sample resume.")
        text = SAMPLE_RESUME
        res = parse_resume(text, "Sample_Resume")
        results.append(res)
        print_results(res)

    if results:
        print(f"\n{'─'*60}")
        print("  SAVING RESULTS")
        print(f"{'─'*60}")
        save_json(results, "resume_results.json")
        save_csv(results,  "resume_results.csv")
        print(f"\n  ✓ {len(results)} resume(s) processed successfully.")
    else:
        print("\n[SKIP] No resumes were processed.")
