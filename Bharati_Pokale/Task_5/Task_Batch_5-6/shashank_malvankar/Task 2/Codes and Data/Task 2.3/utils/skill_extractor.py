skills_db = [

    "python",
    "machine learning",
    "nlp",
    "sql",
    "flask",
    "java",
    "c++",
    "tensorflow",
    "pytorch",
    "data science"
]


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills_db:

        if skill in text:
            found.append(skill)

    return found
