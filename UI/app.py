import streamlit as st
import pickle
import re


# 1. Load the saved vectorizer and model
@st.cache_resource  # This keeps the model loaded in memory so it stays fast
def load_assets():
    with open("../model/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("../sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


try:
    vectorizer, model = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Make sure 'tfidf_vectorizer.pkl' and 'sentiment_model.pkl' are in this directory.")


# 2. Re-define the basic cleaning function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # Remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)  # Remove punctuation & numbers
    return text  # For the app, a light cleanup is usually enough


# 3. Streamlit UI Layout
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a movie review below to see if it's Positive or Negative!")

# Text input box for the user
user_input = st.text_area("Enter your review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Process the input
        cleaned_text = preprocess(user_input)

        # Vectorize the text
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]

        # Display the result beautifully
        st.write("---")
        if prediction == 1:
            st.success("### Result: Positive 👍")
            st.balloons()  # Adds a fun celebration effect!
        else:
            st.error("### Result: Negative 👎")