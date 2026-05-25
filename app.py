import re
import string
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# =====================================================
# LOAD MODEL FILES
# =====================================================
@st.cache_resource

def load_files():
    
    # If you used pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, tfidf, scaler


model, tfidf, scaler = load_files()

# =====================================================
# TEXT PREPROCESSING
# =====================================================
def clean_text(text):
    
    text = str(text)

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def extract_metadata_features(title):

    features = np.array([

        # 1
        title.count('!'),

        # 2
        title.count('?'),

        # 3
        sum(c.isupper() for c in title) / max(len(title), 1),

        # 4
        sum(1 for w in title.split() if w.isupper() and len(w) > 2),

        # 5
        len(title),

        # 6
        len(title.split()),

        # 7
        sum(len(w) for w in title.split()) / max(len(title.split()), 1),

        # 8
        len(re.findall(r'\\d+', title)),

        # 9
        sum(
            w in title.lower()
            for w in [
                'shocking',
                'secret',
                'exposed',
                'breaking',
                'urgent',
                'unbelievable',
                'exclusive',
                'truth',
                'conspiracy',
                'hoax',
                'cover',
                'hidden'
            ]
        )

    ]).reshape(1, -1)

    return features

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_news(headline):

    cleaned = clean_text(headline)

    # TF-IDF vectorization
    text_features = tfidf.transform([cleaned])

    # Metadata features
    metadata_features = extract_metadata_features(headline)

    # Scale metadata features
    metadata_scaled = scaler.transform(metadata_features)

    # Convert to sparse matrix
    metadata_sparse = csr_matrix(metadata_scaled)

    # Combine TF-IDF + metadata features
    final_features = hstack([text_features, metadata_sparse])

    # Prediction
    prediction = model.predict(final_features)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(final_features)[0]
        confidence = np.max(probability)
    else:
        confidence = None

    return prediction, confidence

# =====================================================
# UI
# =====================================================
st.title("📰 Fake News Detection System")

st.markdown(
    """
This application uses Natural Language Processing (NLP) and Machine Learning
models to classify news headlines as either **Real** or **Fake**.
"""
)

headline = st.text_area(
    "Enter a news headline:",
    height=150,
    placeholder="Example: Scientists confirm water discovered on Mars"
)

# =====================================================
# PREDICT BUTTON
# =====================================================
if st.button("Predict"):

    if headline.strip() == "":
        st.warning("Please enter a news headline.")

    else:

        prediction, confidence = predict_news(headline)

        st.subheader("Prediction Result")

        # IMPORTANT:
        # Change labels below if your notebook uses opposite encoding.
        # Current assumption:
        # 0 = Fake
        # 1 = Real

        if prediction == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News Detected")

        if confidence is not None:
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.progress(float(confidence))

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("About Project")

st.sidebar.write(
    """
This fake news detection system was developed as a Big Data Analytics project.

Technologies used:
- Python
- Scikit-learn
- XGBoost
- TF-IDF Vectorization
- Streamlit
- NLP preprocessing
"""
)

st.sidebar.header("Model Information")

st.sidebar.write(
    """
The system predicts whether a news headline is real or fake
using a trained machine learning classifier.
"""
)
