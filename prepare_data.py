import bz2
import pandas as pd
import sqlite3
import os

def prepare_subset():
    print("Data extraction and SQL database creation process is starting. Please wait...")
    
    train_file = 'train.ft.txt.bz2'
    
    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found. Please ensure the file is in the same folder as app.py.")
        return

    data = []
    pos_count = 0
    neg_count = 0
    max_samples = 5000  # Extracting a sample of 10,000 rows (5k positive, 5k negative) for performance
    
    # Reading the .bz2 file line by line without overloading the RAM
    with bz2.BZ2File(train_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8') # Convert byte data to string
            
            # fastText format: __label__1 Review text... OR __label__2 Review text...
            label = line.split(' ')[0]
            review = line[len(label)+1:].strip()
            
            if label == '__label__1' and neg_count < max_samples:
                data.append(('Negative', review))
                neg_count += 1
            elif label == '__label__2' and pos_count < max_samples:
                data.append(('Positive', review))
                pos_count += 1
                
            # Stop reading once we reach our target sample size
            if pos_count == max_samples and neg_count == max_samples:
                break
                
    # Convert extracted data to a Pandas DataFrame
    df = pd.DataFrame(data, columns=['Sentiment', 'Review'])
    
    print(f"A total of {len(df)} reviews have been successfully read. Writing to SQL database...")
    
    # Establish SQLite database connection (Creates the file if it doesn't exist)
    conn = sqlite3.connect('amazon_reviews.db')
    
    # Export the data to SQL as a table
    df.to_sql('reviews', conn, if_exists='replace', index=False)
    conn.close()
    
    print("✅ Process Completed! Data has been successfully saved to a SQL database named 'amazon_reviews.db'.")

if __name__ == "__main__":
    prepare_subset()