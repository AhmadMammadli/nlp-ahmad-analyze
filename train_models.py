import sqlite3
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

print("1. Loading data from SQL Database...")
conn = sqlite3.connect('amazon_reviews.db')
df = pd.read_sql_query("SELECT * FROM reviews", conn)
conn.close()

print("2. Cleaning text data...")
df['Clean_Review'] = df['Review'].apply(clean_text)

print("3. Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['Clean_Review'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Logistic Regression ---
print("\n4. Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print(classification_report(y_test, log_pred))

# --- Model 2: Random Forest ---
print("\n5. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(classification_report(y_test, rf_pred))

# --- Model 3: Naive Bayes ---
print("\n6. Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print(classification_report(y_test, nb_pred))

print("\n7. Saving models and vectorizer...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(log_model, 'log_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')

print("✅ Process Completed! All 3 models trained and saved.")