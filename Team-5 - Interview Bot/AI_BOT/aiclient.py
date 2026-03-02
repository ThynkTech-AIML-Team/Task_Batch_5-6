from google import genai
from config import API_KEY, MIN_QUESTIONS

# Gemini Client
client = genai.Client(api_key=API_KEY)


def generate_questions(domain):

    prompt = f"""
You are a professional HR interviewer.

Generate exactly {MIN_QUESTIONS} interview questions.

Domain: {domain}

Rules:

- Mix technical and HR questions
- Professional interview style
- Short and clear
- One question per line
- Do not number questions
- Do not add explanation

Return only questions.
"""


    try:

        response = client.models.generate_content(

            model="gemini-2.0-flash",

            contents=prompt

        )


        text = response.text


        questions = []


        for line in text.split("\n"):

            line = line.strip()


            # Remove numbering like "1." or "2)"
            if line[:2].isdigit():

                line = line[2:].strip()

            if len(line) > 8:

                questions.append(line)


        return questions[:MIN_QUESTIONS]


    except Exception as e:

        print("Gemini Error:", e)


        # Backup questions
        return [

            "Tell me about yourself",

            "Explain one project you built",

            "What are your strengths",

            "Why should we hire you",

            "What are your career goals"

        ]