import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Ensure 'punkt' is downloaded for tokenization
nltk.download('punkt')

def summarize_text(text, sentence_count=3):
    print("--- Original Text Length:", len(text.split()), "words ---")
    
    # 1. Parse the text
    # 'english' is the language for the tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # 2. Initialize the Summarizer
    # LSA (Latent Semantic Analysis) is a common summarization algorithm
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    
    # 3. Generate Summary
    print(f"\n--- Summary ({sentence_count} sentences) ---")
    summary = summarizer(parser.document, sentence_count)
    
    for sentence in summary:
        print(f"- {sentence}")

# Example Data (A short article about AI)
input_text = """
'The Last Frontier' is a breathtaking cinematic achievement that successfully blends high-stakes science fiction with a deeply personal human story. Directed by Elena Vance, the film follows a group of explorers who travel through a newly discovered wormhole to find a home for humanity. The narrative is structured around the emotional journey of Captain Miller, whose internal conflict between duty and family provides the movie's emotional heartbeat. 

Visually, the film is a triumph, featuring practical effects and stunning cinematography that make the vastness of space feel both terrifying and beautiful. The musical score, composed by Hans Zimmer, further elevates the tension, using low-frequency pulses that mimic a ticking clock. Critics have praised the film for its scientific accuracy, as the production team consulted with leading physicists to ensure the black hole visuals were as realistic as possible.

However, the movie is not without its flaws, as some viewers have complained about the nearly three-hour runtime and a middle act that feels slightly repetitive. Despite these minor pacing issues, the screenplay manages to deliver a powerful message about the resilience of the human spirit in the face of extinction. The final sequence is particularly moving, tying together multiple plot threads in a way that is both surprising and satisfying.

Ultimately, 'The Last Frontier' is a must-watch for any fan of the genre, offering a rare combination of intellectual depth and visual spectacle. It is currently breaking box office records across the globe and is expected to be a frontrunner during the upcoming awards season. While it may require patience from the audience, the payoff is more than worth the investment, cementing Vance's reputation as one of the most visionary directors working today."""

if __name__ == "__main__":
    summarize_text(input_text, sentence_count=2)