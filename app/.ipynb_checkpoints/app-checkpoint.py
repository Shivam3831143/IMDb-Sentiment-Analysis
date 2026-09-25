import streamlit as st
import joblib
import re
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# Load NLP resources
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# Text preprocessing function
# --------------------------------------------------

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into words
    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------------------------
# Load trained model and TF-IDF vectorizer
# --------------------------------------------------

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "svm_sentiment_model.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# Application Interface
# --------------------------------------------------

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review "
)


review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder="Example: This movie was absolutely fantastic..."
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        # Clean the review
        cleaned_review = clean_text(review)

        # Convert review into TF-IDF features
        review_tfidf = tfidf.transform([cleaned_review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)

        # Display result
        if prediction[0] == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")