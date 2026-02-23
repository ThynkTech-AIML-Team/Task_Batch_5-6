import nltk
nltk.download('punkt')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

text = """
An unmanned aerial vehicle (UAV) is defined as a "powered, aerial vehicle that does not carry a human operator, uses aerodynamic forces to provide vehicle lift, can fly autonomously or be piloted remotely, can be expendable or recoverable, and can carry a lethal or nonlethal payload". UAV is a term that is commonly applied to military use cases. Missiles with warheads are generally not considered UAVs because the vehicle itself is a munition, but certain types of propeller-based missile are often called "kamikaze drones" by the public and media. Also, the relation of UAVs to remote controlled model aircraft is unclear in some jurisdictions. The US FAA now defines any unmanned flying craft as a UAV regardless of weight. Similar terms are remotely piloted aircraft (RPA) and remotely piloted aerial vehicle (RPAV).
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 3)

print("Summary:\n")
for sentence in summary:
    print(sentence)
