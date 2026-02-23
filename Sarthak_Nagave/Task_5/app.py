import streamlit as st
import numpy as np
import cv2
import pandas as pd
import joblib
import time
from tensorflow.keras.models import load_model
from PIL import Image

#CONFIG
st.set_page_config(
    page_title="AI Digit Recognition",
    layout="wide"
)

#LOAD MODELS
cnn_model = load_model("mnist_cnn_model.h5")
lr_model = joblib.load("mnist_lr_model.pkl")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

st.sidebar.markdown("## <i class='fa-solid fa-compass'></i> Navigation", unsafe_allow_html=True)
page = st.sidebar.selectbox("", ["Predict", "Analytics"])
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

if theme == "Dark":
    background = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
    card_color = "rgba(255,255,255,0.06)"
    text_color = "white"
else:
    background = "linear-gradient(135deg, #f5f7fa, #c3cfe2)"
    card_color = "white"
    text_color = "black"

st.markdown(f"""
<style>
.stApp {{
    background: {background};
}}

.card {{
    background: {card_color};
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}}

.title {{
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:{text_color};
}}

.sub {{
    text-align:center;
    color:{text_color};
    opacity:0.8;
}}

.metric {{
    font-size:20px;
    font-weight:600;
}}
</style>
""", unsafe_allow_html=True)

# PREDICT PAGE 

if page == "Predict":

    st.markdown("<div class='title'><i class='fa-solid fa-brain'></i> AI Digit Recognition</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Upload MNIST-style grayscale image (28x28)</div><br>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")
        st.image(image, width=220)

        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (28, 28))
        img_norm = img_resized / 255.0

        img_cnn = img_norm.reshape(1, 28, 28, 1)
        img_lr = img_norm.reshape(1, 784)

        if st.button("Predict Digit", type="primary"):

            with st.spinner("Processing..."):
                time.sleep(1)

                cnn_pred = cnn_model.predict(img_cnn)
                cnn_digit = int(np.argmax(cnn_pred))
                cnn_conf = float(np.max(cnn_pred) * 100)

                lr_digit = int(lr_model.predict(img_lr)[0])
                lr_conf = float(np.max(lr_model.predict_proba(img_lr)) * 100)

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"CNN Prediction: {cnn_digit}")
                st.write(f"Confidence: {cnn_conf:.2f}%")

            with col2:
                st.info(f"Logistic Regression: {lr_digit}")
                st.write(f"Confidence: {lr_conf:.2f}%")

            st.subheader("Probability Distribution")
            st.bar_chart(cnn_pred[0])

            st.session_state.history.append({
                "CNN Prediction": cnn_digit,
                "CNN Confidence (%)": round(cnn_conf, 2),
                "LR Prediction": lr_digit,
                "LR Confidence (%)": round(lr_conf, 2)
            })

            st.audio("https://www.soundjay.com/buttons/sounds/button-3.mp3")

    st.markdown("</div>", unsafe_allow_html=True)


# ANALYTICS PAGE 

elif page == "Analytics":

    st.markdown("<div class='title'><i class='fa-solid fa-chart-line'></i> Analytics Dashboard</div><br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.metric("Total Predictions", len(df))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CNN Distribution")
            st.bar_chart(df["CNN Prediction"].value_counts())

        with col2:
            st.subheader("LR Distribution")
            st.bar_chart(df["LR Prediction"].value_counts())

        st.subheader("Prediction History")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

    else:
        st.info("No predictions yet.")

    st.markdown("</div>", unsafe_allow_html=True)