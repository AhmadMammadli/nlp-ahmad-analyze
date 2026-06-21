import streamlit as st
import pandas as pd
import sqlite3
import joblib
import re

# Page configurations
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Main Header
st.title("🎙️ Amazon Alexa Reviews - Sentiment Analysis Dashboard")

# Helper function to clean text (Must match the training phase)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load Models and Vectorizer
@st.cache_resource
def load_models():
    vec = joblib.load('tfidf_vectorizer.pkl')
    log_reg = joblib.load('log_model.pkl')
    rf_clf = joblib.load('rf_model.pkl')
    return vec, log_reg, rf_clf

# Load Data from SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect('amazon_reviews.db')
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    return df

try:
    # Initialize components
    vectorizer, log_model, rf_model = load_models()
    df = load_data()
    
    # Dashboard Layout (Two Columns)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Database Overview")
        st.write(f"Total records in SQL Database: **{len(df)}**")
        st.dataframe(df.sample(10, random_state=42), use_container_width=True)
        
        st.subheader("📈 Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
    with col2:
        st.subheader("🤖 Test the AI Models")
        st.write("Type a custom review below to see how our trained models classify it.")
        
        user_input = st.text_area("Enter your review here:", "Alexa is the best smart speaker I have ever used!")
        model_choice = st.selectbox("Choose a Machine Learning Model:", ["Logistic Regression", "Random Forest"])
        
        if st.button("Analyze Sentiment"):
            if user_input:
                # 1. Clean the text
                cleaned_input = clean_text(user_input)
                # 2. Vectorize (Convert words to numbers)
                vectorized_input = vectorizer.transform([cleaned_input])
                
                # 3. Predict Sentiment
                if model_choice == "Logistic Regression":
                    prediction = log_model.predict(vectorized_input)[0]
                else:
                    prediction = rf_model.predict(vectorized_input)[0]
                    
                # 4. Display Results
                if prediction == "Positive":
                    st.success(f"**Prediction Result:** {prediction} 🤩")
                else:
                    st.error(f"**Prediction Result:** {prediction} 😠")
            else:
                st.warning("Please enter a review to analyze.")
                
except Exception as e:
    st.error(f"Error loading system components. Please ensure you have run 'train_models.py' first. Details: {e}")