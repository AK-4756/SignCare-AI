import pandas as pd
import os

csv_path = os.path.join('Data', 'alphabet_data.csv')

if os.path.exists(csv_path):
    # 1. Load the data
    df = pd.read_csv(csv_path)
    
    # 2. Define which letters you want to REMOVE
    # You can add more letters to this list later if they get "confused"
    letters_to_remove = ['P']
    
    print(f"📊 Rows before cleaning: {len(df)}")
    
    # 3. Filter out those letters
    df_cleaned = df[~df['label'].isin(letters_to_remove)]
    
    # 4. Save the clean data back to the same file
    df_cleaned.to_csv(csv_path, index=False)
    
    print(f"✅ Successfully removed letters: {letters_to_remove}")
    print(f"📊 Rows after cleaning: {len(df_cleaned)}")
else:
    print("❌ Error: CSV file not found!")