import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Load the trained model and stopwords
try:
    model = joblib.load("models/fake_news_pipeline.joblib")
    nlp = spacy.load("en_core_web_sm")
    list1 = nlp.Defaults.stop_words
    list2 = stopwords.words('english')
    Stopwords = set(list1) | set(list2)
    lemma = WordNetLemmatizer()
except Exception as e:
    st.error(f"Error loading model or resources: {e}")
    st.stop()

def clean_text(text):
    """Cleans the input text for prediction."""
    string = ""
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#!@$%^&*{}?.,:]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)

    for word in text.split():
        if word not in Stopwords:
            string += lemma.lemmatize(word) + " "

    return string

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a news article below to check if it's likely real or fake.")

user_input = st.text_area("News Article Text", height=200)

if st.button("Check"):
    if user_input:
        with st.spinner("Analyzing..."):
            cleaned_input = clean_text(user_input)
            prediction = model.predict([cleaned_input])
            probability = model.predict_proba([cleaned_input])

            if prediction[0] == 1:
                st.error(f"This looks like FAKE news (Probability: {probability[0][1]:.2f})")
            else:
                st.success(f"This looks like REAL news (Probability: {probability[0][0]:.2f})")
    else:
        st.warning("Please enter some news text to analyze.")
