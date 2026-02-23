# Computer Vision Task 5

## Dataset: MNIST (70,000 handwritten digits, 10 classes)

## Models Used
| Model              | Test Accuracy | Notes |
|--------------------|---------------|-------|
| Logistic Regression| ~0.92        | Traditional ML, simple but less accurate for images. |
| CNN                | ~0.98        | Deep learning, better feature extraction. |

## Results
- CNN outperforms LR due to convolutional layers.
- See notebook for confusion matrices and accuracy graphs.

## Deployment
Run `streamlit run app.py` for digit recognition app.