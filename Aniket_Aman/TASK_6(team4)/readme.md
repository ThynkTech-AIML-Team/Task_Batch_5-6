#  COCO Image Captioner (Computer Vision & NLP)

## 🚀 Project Overview
This project implements a fast, efficient deep learning system that automatically generates descriptive English captions for images. 

The system leverages a hybrid CNN-RNN architecture to extract visual features and sequence text, allowing the AI to "see" an image and accurately describe its contents.

---

## 🌐 Live Demo
You can test out the interactive web application here: 
👉 **[COCO Image Captioner Live App](https://huggingface.co/spaces/golulog/coco-image-captioner)**

*(Note: This is a temporary Gradio live link and requires the host environment to be actively running).*

---

## 🎯 Objective
Generate accurate, descriptive captions for unseen images using visual feature extraction and sequential text generation, complete with an interactive web interface for real-time testing.

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Pre-trained MobileNetV2
- Natural Language Toolkit (NLTK)
- Gradio (Web UI)
- Pandas, NumPy

---

## 📊 Methodology

### 1️⃣ Data Processing
- Extracted and parsed MS COCO `val2017` dataset annotations.
- Implemented a targeted 5,000-image subset for rapid training.
- Cleaned text data (lowercase, punctuation removal, startseq/endseq addition).
- Tokenized text into numerical sequences (7,324 word vocabulary).

### 2️⃣ Image Feature Extraction
- Utilized pre-trained MobileNetV2 (CNN) without the top classification layer.
- Extracted 1,280-dimensional dense feature vectors for every image in the dataset.
- Saved extracted features for optimized, rapid model training.

### 3️⃣ Deep Captioning Model Architecture
- **Image Pathway:** Dropout and Dense layers to process MobileNetV2 features.
- **Text Pathway:** Word Embeddings and an LSTM network to capture language sequences.
- **Merge:** Combined visual and text pathways to predict the next sequential word using categorical crossentropy.

### 4️⃣ Training & Evaluation
- Custom Python data generator for memory-efficient batching.
- 90/10 Train-Test split.
- Evaluated generated captions against ground-truth MS COCO data using BLEU metrics.

---

## 📈 Performance Metrics
- **BLEU-1 Score:** 0.6113
- **BLEU-2 Score:** 0.4283

The CNN-LSTM architecture successfully learned the mapping between complex image features and English vocabulary structure.

---

## 📊 Visualization & UI
- **Gradio Web Application:** Built an interactive, local web interface.
- Users can upload any custom image and receive an AI-generated caption in real-time.

---

## 💾 Deployment Ready
- Trained model saved (`fast_coco_model.h5`)
- NLP Tokenizer and extracted features saved (`tokenizer.pkl`, `features.pkl`, `subset_mapping.pkl`)
- Gradio script ready for local hosting or public link sharing.

---

## 👤 Team 4 - Aniket Aman, Ayush Mansukh Singh, KrishnaKumar Harindra Yadav, Priyal Mangesh Mandloi, Roshan Shailesh Bilsore, Ansh Ajay Agarwal



CNN-RNN Architecture, MobileNetV2 Feature Extraction & Web UI Integration

