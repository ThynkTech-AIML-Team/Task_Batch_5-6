import streamlit as st
import spacy
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Resume Parser", page_icon="📄")

st.title("📄 AI Resume Parser")
st.write("Upload a resume PDF and extract structured information using NLP.")

uploaded_file = st.file_uploader("Upload Resume (PDF format only)", type=["pdf"])

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

if uploaded_file is not None:

    resume_text = extract_text_from_pdf(uploaded_file)
    doc = nlp(resume_text)

    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]

    candidate_name = "Not Found"

    top_section = " ".join(lines[:10])
    doc_top = nlp(top_section)

    for ent in doc_top.ents:
        if ent.label_ == "PERSON":

            cleaned_name = ent.text

            cleaned_name = re.sub(r"email.*", "", cleaned_name, flags=re.IGNORECASE)
            cleaned_name = re.sub(r"phone.*", "", cleaned_name, flags=re.IGNORECASE)
            cleaned_name = re.sub(r"\d+", "", cleaned_name)

            cleaned_name = cleaned_name.strip()

            words = cleaned_name.split()
            words = [w for w in words if w.isalpha()]

            if 1 < len(words) <= 4:
                candidate_name = " ".join(words)
                break

    if candidate_name == "Not Found" and lines:
        possible_name = lines[0]
        possible_name = re.sub(r"\d+", "", possible_name)
        words = [w for w in possible_name.split() if w.isalpha()]
        if 1 < len(words) <= 4:
            candidate_name = " ".join(words)

    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+"
    emails = list(set(re.findall(email_pattern, resume_text)))

    phone_pattern = r"\+?\d[\d\s-]{8,14}"
    phones_raw = re.findall(phone_pattern, resume_text)

    phones = []
    for phone in phones_raw:
        cleaned = re.sub(r"\s+", " ", phone).strip()
        phones.append(cleaned)

    phones = list(set(phones))

    organizations = []

    ignore_words = [
        "data science", "machine learning",
        "deep learning", "css", "javascript"
    ]

    for ent in doc.ents:
        if ent.label_ == "ORG":
            if ent.text.lower() not in ignore_words:
                organizations.append(ent.text)

    organizations = list(set(organizations))

    education_keywords = [
        "bachelor", "master", "phd",
        "b.e", "btech", "mtech",
        "engineering", "university",
        "college"
    ]

    education_details = []

    for line in lines:
        for keyword in education_keywords:
            if keyword.lower() in line.lower():
                education_details.append(line)

    education_details = list(set(education_details))

    skill_database = [
        "python", "java", "c++", "machine learning",
        "deep learning", "nlp", "data science",
        "sql", "html", "css", "javascript",
        "tensorflow", "pytorch", "react",
        "nodejs", "power bi", "excel"
    ]

    found_skills = set()

    for skill in skill_database:
        if re.search(r"\b" + re.escape(skill) + r"\b", resume_text.lower()):
            found_skills.add(skill)

    candidate_phrases = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower()
        if 1 <= len(phrase.split()) <= 3:
            if not any(char.isdigit() for char in phrase):
                candidate_phrases.append(phrase)

    candidate_phrases = list(set(candidate_phrases))

    skill_embeddings = embedding_model.encode(skill_database)
    phrase_embeddings = embedding_model.encode(candidate_phrases)

    for i, phrase_emb in enumerate(phrase_embeddings):

        similarities = cosine_similarity(
            [phrase_emb],
            skill_embeddings
        )[0]

        max_sim = np.max(similarities)
        best_match = skill_database[np.argmax(similarities)]

        if max_sim > 0.55:
            found_skills.add(best_match)

    found_skills = list(found_skills)

    st.subheader("Parsed Information")

    st.markdown("###  Name")
    st.write(candidate_name)

    st.markdown("###  Email")
    st.write(emails if emails else "Not Found")

    st.markdown("###  Phone")
    st.write(phones if phones else "Not Found")

    st.markdown("###  Organizations")
    st.write(organizations if organizations else "Not Found")

    st.markdown("###  Education")
    st.write(education_details if education_details else "Not Found")

    st.markdown("###  Skills (AI Detected)")
    st.write(found_skills if found_skills else "Not Found")

    st.success("Resume parsing completed successfully!")
