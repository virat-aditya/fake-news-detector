import streamlit as st
import joblib
import re
import os
import sys
import subprocess
from pathlib import Path

# Download NLTK data
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    with st.spinner("Downloading language resources..."):
        nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    with st.spinner("Downloading language resources..."):
        nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    with st.spinner("Downloading language resources..."):
        nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load spaCy model
import spacy

@st.cache_resource
def load_spacy_model():
    """Load spaCy model with error handling."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to install spaCy model: {e}")
            st.stop()

@st.cache_resource
def load_model_and_resources():
    """Load the trained model and initialize resources."""
    try:
        # Check if model file exists
        model_path = "models/fake_news_pipeline.joblib"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            st.info("Please ensure the model file is uploaded to your repository in the 'models' folder.")
            st.stop()
        
        # Load model
        model = joblib.load(model_path)
        
        # Load spaCy model
        nlp = load_spacy_model()
        
        # Initialize stopwords
        try:
            list1 = nlp.Defaults.stop_words
            list2 = stopwords.words('english')
            combined_stopwords = set(list1) | set(list2)
        except Exception as e:
            st.warning(f"Error loading stopwords: {e}. Using default English stopwords.")
            combined_stopwords = set(stopwords.words('english'))
        
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        return model, combined_stopwords, lemmatizer
        
    except Exception as e:
        st.error(f"Error loading model or resources: {e}")
        st.info("Please check that all required files are in your repository and requirements are properly installed.")
        st.stop()

def clean_text(text, stopwords_set, lemmatizer):
    """Cleans the input text for prediction."""
    if not text or not isinstance(text, str):
        return ""
    
    string = ""
    text = text.lower()
    
    # Contract expansions
    contractions = {
        r"i'm": "i am",
        r"he's": "he is",
        r"she's": "she is", 
        r"that's": "that is",
        r"what's": "what is",
        r"where's": "where is",
        r"\'ll": " will",
        r"\'ve": " have",
        r"\'re": " are",
        r"\'d": " would",
        r"won't": "will not",
        r"can't": "cannot"
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(contraction, expansion, text)
    
    # Clean punctuation and special characters
    text = re.sub(r"[-()\"#!@$%^&*{}?.,:]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    
    # Remove stopwords and lemmatize
    for word in text.split():
        if word and word not in stopwords_set:
            string += lemmatizer.lemmatize(word) + " "
    
    return string.strip()

# Load resources
model, Stopwords, lemma = load_model_and_resources()

# Streamlit UI
st.title("🔍 Fake News Detector")
st.write("Enter a news article below to check if it's likely real or fake.")

# Add some styling
st.markdown("""
<style>
.stTextArea textarea {
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Input section
user_input = st.text_area(
    "📰 News Article Text", 
    height=200,
    placeholder="Paste your news article here..."
)

# Analysis section
col1, col2 = st.columns([1, 4])

with col1:
    analyze_button = st.button("🔎 Analyze", type="primary")

with col2:
    if st.button("🗑️ Clear"):
        st.experimental_rerun()

if analyze_button:
    if user_input and user_input.strip():
        with st.spinner("🤖 Analyzing article..."):
            try:
                # Clean the input
                cleaned_input = clean_text(user_input, Stopwords, lemma)
                
                if not cleaned_input:
                    st.warning("⚠️ The text couldn't be processed properly. Please try with different text.")
                else:
                    # Make prediction
                    prediction = model.predict([cleaned_input])
                    probability = model.predict_proba([cleaned_input])
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    if prediction[0] == 1:
                        st.error(f"🚨 **FAKE NEWS DETECTED**")
                        st.error(f"Confidence: {probability[0][1]:.1%}")
                        st.markdown("⚠️ This article shows characteristics commonly found in fake news.")
                    else:
                        st.success(f"✅ **APPEARS TO BE REAL NEWS**")
                        st.success(f"Confidence: {probability[0][0]:.1%}")
                        st.markdown("✓ This article appears to have characteristics of legitimate news.")
                    
                    # Add confidence bar
                    fake_prob = probability[0][1]
                    real_prob = probability[0][0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real News Probability", f"{real_prob:.1%}")
                    with col2:
                        st.metric("Fake News Probability", f"{fake_prob:.1%}")
                        
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.info("Please try again or contact support if the issue persists.")
    else:
        st.warning("⚠️ Please enter some news text to analyze.")

# Add footer with disclaimer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
<p><strong>Disclaimer:</strong> This tool provides predictions based on machine learning analysis. 
Always verify news from multiple reliable sources before making judgments.</p>
</div>
""", unsafe_allow_html=True)
