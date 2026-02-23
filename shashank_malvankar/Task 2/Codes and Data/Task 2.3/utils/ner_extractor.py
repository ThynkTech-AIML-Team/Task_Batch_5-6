import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):

    doc = nlp(text)

    names = []
    organizations = []

    for ent in doc.ents:

        if ent.label_ == "PERSON":
            names.append(ent.text)

        if ent.label_ == "ORG":
            organizations.append(ent.text)

    email = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone = re.findall(
        r"\+?\d[\d -]{8,12}\d",
        text
    )

    return {

        "Name": names,
        "Organizations": organizations,
        "Email": email,
        "Phone": phone
    }