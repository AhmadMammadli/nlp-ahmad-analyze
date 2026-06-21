# 🎙️ Amazon Alexa Reviews - Sentiment Analysis Dashboard

## Project Overview
This project is an end-to-end Natural Language Processing (NLP) system designed to automatically classify Amazon customer reviews into Positive or Negative sentiments. It features a complete pipeline including data extraction from massive datasets, SQL database storage, machine learning model training, and an interactive web dashboard.

## Key Features
* **Efficient Data Processing:** Reads and processes large-scale compressed datasets (`.bz2`) using stream processing to optimize memory usage.
* **SQL Database Integration:** Stores processed text data and sentiment labels into a lightweight SQLite database (`amazon_reviews.db`) for robust data management.
* **Machine Learning Pipeline:** Utilizes TF-IDF vectorization for text representation and trains two distinct models for comparison: **Logistic Regression** and **Random Forest Classifier**.
* **Interactive Dashboard:** A Streamlit-based web application that visualizes data distribution and allows users to input custom reviews for real-time sentiment prediction.

## Project Structure
* `prepare_data.py`: Extracts a balanced subset of data from the raw `.bz2` file and initializes the SQLite database.
* `train_models.py`: Connects to the SQL database, preprocesses the text, trains the ML models, and exports them as `.pkl` files.
* `app.py`: The main Streamlit application script that loads the database and models to serve the user interface.

## How to Run the Project
1. **Prepare the Data & Database:**
   ```bash
   python prepare_data.py
2. **Train the ML Models:**
   ```bash
   python train_models.py
3. **Launch the Dashboard:**
   ```bash
   python -m streamlit run app.py
   
## Dashboard Screenshot
![Sentiment Analysis Dashboard](dashboard.png)