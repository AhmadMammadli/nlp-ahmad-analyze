import sqlite3
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def clean_text(text):
    # Metni küçük harflere çevir ve sadece İngilizce harfleri tut
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

print("1. Loading data from SQL Database...")
conn = sqlite3.connect('amazon_reviews.db')
df = pd.read_sql_query("SELECT * FROM reviews", conn)
conn.close()

print("2. Cleaning text data...")
df['Clean_Review'] = df['Review'].apply(clean_text)

print("3. Vectorizing text (TF-IDF)...")
# TF-IDF ile kelimeleri sayısallaştır ve yaygın İngilizce dolgu kelimelerini (stop words) at
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['Clean_Review'])
y = df['Sentiment']

# Veriyi %80 Eğitim, %20 Test olarak ikiye ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training Model 1: Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print(f"   -> Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.2f}")

print("5. Training Model 2: Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"   -> Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")

print("6. Saving models and vectorizer...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(log_model, 'log_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')

print("✅ Process Completed! All models are successfully trained and saved as .pkl files.")