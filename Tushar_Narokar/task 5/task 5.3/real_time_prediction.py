import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

def main():
    
    print("Loading InceptionV3 model...")
    model = InceptionV3(weights='imagenet')
    print("Model loaded.")

   
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time prediction... Press 'q' to quit.")

    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        
        resized_frame = cv2.resize(frame, (299, 299))
        img_array = np.expand_dims(resized_frame, axis=0)
        img_array = preprocess_input(img_array)

       
        predictions = model.predict(img_array, verbose=0)
        decoded_preds = decode_predictions(predictions, top=3)[0]

        
        for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
            text = f"{label}: {prob*100:.2f}%"
            cv2.putText(frame, text, (10, 30 + (i * 30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

       
        cv2.imshow('InceptionV3 Real-time Classification', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")

if __name__ == "__main__":
    main()
