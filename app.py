import streamlit as st
import pandas as pd
import sqlite3
import joblib
import re
from datetime import datetime
from collections import Counter

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("🎙️ Amazon Alexa Reviews - Advanced Sentiment Analysis")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource
def load_models():
    vec = joblib.load('tfidf_vectorizer.pkl')
    log_reg = joblib.load('log_model.pkl')
    rf_clf = joblib.load('rf_model.pkl')
    nb_clf = joblib.load('nb_model.pkl')
    return vec, log_reg, rf_clf, nb_clf

def init_db():
    conn = sqlite3.connect('amazon_reviews.db')
    cursor = conn.cursor()
    # Create table for real-time predictions if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            timestamp TEXT,
            review_text TEXT,
            predicted_sentiment TEXT,
            confidence_score REAL,
            model_used TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(review, sentiment, confidence, model_name):
    conn = sqlite3.connect('amazon_reviews.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_logs (timestamp, review_text, predicted_sentiment, confidence_score, model_used)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), review, sentiment, confidence, model_name))
    conn.commit()
    conn.close()

try:
    init_db()
    vectorizer, log_model, rf_model, nb_model = load_models()
    
    conn = sqlite3.connect('amazon_reviews.db')
    df_original = pd.read_sql_query("SELECT * FROM reviews LIMIT 5000", conn)
    df_logs = pd.read_sql_query("SELECT * FROM prediction_logs", conn)
    conn.close()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🤖 Test the AI Models")
        user_input = st.text_area("Enter your review here:", "Alexa is very helpful and fast!")
        model_choice = st.selectbox("Choose a Model:", ["Logistic Regression", "Random Forest", "Naive Bayes"])
        
        if st.button("Analyze Sentiment"):
            if user_input:
                cleaned_input = clean_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])
                
                if model_choice == "Logistic Regression":
                    pred = log_model.predict(vectorized_input)[0]
                    prob = log_model.predict_proba(vectorized_input).max()
                elif model_choice == "Random Forest":
                    pred = rf_model.predict(vectorized_input)[0]
                    prob = rf_model.predict_proba(vectorized_input).max()
                else:
                    pred = nb_model.predict(vectorized_input)[0]
                    prob = nb_model.predict_proba(vectorized_input).max()
                
                log_prediction(user_input, pred, prob, model_choice)
                
                if pred == "Positive":
                    st.success(f"**Sentiment:** {pred} (Confidence: {prob:.2f})")
                else:
                    st.error(f"**Sentiment:** {pred} (Confidence: {prob:.2f})")
                st.info("✅ Result logged to SQL Database with timestamp!")
                
    with col2:
        st.subheader("📈 Statistical Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Sentiment Dist", "Word Frequency", "Prediction Logs"])
        
        with tab1:
            st.bar_chart(df_original['Sentiment'].value_counts())
            
        with tab2:
            all_words = ' '.join(df_original['Review'].apply(clean_text).tolist()).split()
            word_counts = Counter(all_words).most_common(15)
            df_words = pd.DataFrame(word_counts, columns=['Word', 'Frequency']).set_index('Word')
            st.bar_chart(df_words)
            
        with tab3:
            st.write("Recent Database Logs (Timestamps & Confidence Scores)")
            st.dataframe(df_logs.tail(5))
            
except Exception as e:
    st.error(f"System Error. Details: {e}")