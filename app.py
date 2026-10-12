import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Streamlit Configuration
# ------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# Project Directory
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------
# Load Saved Files
# ------------------------------
@st.cache_resource
def load_resources():
    model_path = BASE_DIR / "lstm_model.h5"
    tokenizer_path = BASE_DIR / "tokenizer.pkl"
    max_len_path = BASE_DIR / "max_len.pkl"

    # Check required files
    for file_path in [model_path, tokenizer_path, max_len_path]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load model
    model = load_model(model_path)

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load maximum sequence length
    with open(max_len_path, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


# ------------------------------
# Prediction Function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    predictions = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(predictions))

    # Find the predicted word efficiently
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None


# ------------------------------
# Load Resources
# ------------------------------
try:
    model, tokenizer, max_len = load_resources()
except Exception as e:
    st.error(f"Failed to load model resources: {e}")
    st.stop()


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Next Word Prediction")

st.write(
    "Enter a sentence, and the trained LSTM model "
    "will predict the next word."
)

user_input = st.text_input(
    "✍️ Enter text",
    placeholder="Type a sentence here..."
)

if st.button("Predict Next Word", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Predicting the next word..."):
            next_word = predict_next_word(user_input)

        if next_word:
            st.success(f"Predicted Next Word: **{next_word}**")
        else:
            st.warning(
                "Could not predict a next word. "
                "Try entering a different sentence."
            )

st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")