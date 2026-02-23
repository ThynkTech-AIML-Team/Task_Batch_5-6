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
The launch of 5G services in India marks a transformative chapter in the country's digital journey. Officially inaugurated by Prime Minister Narendra Modi in October 2022 at the India Mobile Congress, 5G technology promises to revolutionize how Indians connect, communicate, and consume content. Unlike its predecessors, 5G is not merely about faster internet speeds; it is a leap forward in connectivity that offers ultra-low latency, massive machine-type communications, and greater reliability. This technological upgrade is expected to be a critical enabler for India's goal of becoming a $5 trillion economy by empowering sectors like healthcare, agriculture, education, and manufacturing.
Leading the charge in this rollout are India's two major telecom giants: Reliance Jio and Bharti Airtel. Reliance Jio has opted for a Standalone (SA) 5G architecture, which operates independently of the existing 4G infrastructure. This ambitious approach allows Jio to offer "True 5G" capabilities, including network slicing and ultra-low latency, which are essential for advanced applications like autonomous vehicles and remote surgery. On the other hand, Bharti Airtel has deployed a Non-Standalone (NSA) architecture, which utilizes existing 4G infrastructure to deliver 5G services. This strategy has allowed Airtel to roll out services rapidly across major cities, ensuring that customers with 5G-enabled devices can experience enhanced speeds immediately.
The impact of 5G in India extends far beyond high-speed downloads and buffer-free streaming. In the healthcare sector, high-speed connectivity enables telemedicine and remote diagnostics, bridging the gap between urban hospitals and rural patients. For agriculture, 5G-powered IoT (Internet of Things) devices can monitor soil health, weather conditions, and crop growth in real-time, helping farmers increase yields and reduce wastage. Furthermore, the education sector stands to benefit from immersive learning experiences through Virtual Reality (VR) and Augmented Reality (AR), making quality education accessible to students in the remotest corners of the country.
However, the rollout is not without its challenges. The infrastructure cost for 5G is immense, requiring the installation of extensive fiber-optic networks and thousands of small cells to ensure consistent coverage. Telecom operators are under significant financial pressure to monetize these services while keeping tariffs affordable for the price-sensitive Indian market. Additionally, the device ecosystem is still maturing; while 5G smartphone shipments are growing, a large portion of the population still uses 4G or even 2G devices.
Despite these hurdles, the speed of adoption has been unprecedented. Within just a year of its launch, India became one of the fastest countries to roll out 5G sites, surpassing many developed nations in deployment speed. The government continues to support this ecosystem through Production Linked Incentive (PLI) schemes to boost local manufacturing of telecom gear. As the network stabilizes and coverage expands to rural hinterlands, 5G is set to become the backbone of Digital India, driving innovation and bridging the digital divide for millions of citizens.
"""

if __name__ == "__main__":
    summarize_text(input_text, sentence_count=4)