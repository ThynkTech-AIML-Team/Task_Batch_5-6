# CIFAR-10 Image Classification Project

## Objective
The goal of this project is to build and compare image classification models using both traditional machine learning (Logistic Regression) and deep learning (CNN) techniques on the CIFAR-10 dataset.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Model Comparison
| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| Logistic Regression | ~35-40% | Simple, fast to train | Linear model, poor at capturing spatial hierarchy |
| CNN | ~70-80% | High accuracy, captures spatial features | Computationally intensive, longer training time |

## How to Run
1. Create a virtual environment: `py -3.11 -m venv venv`
2. Activate it: `.\venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Jupyter Notebook: `jupyter notebook`

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Training vs Validation Accuracy Graph
