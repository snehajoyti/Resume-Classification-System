import os
import docx
import pandas as pd

# Function to extract text from docx
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return " ".join([para.text for para in doc.paragraphs])

data = []

# Folder path
folder_path = r"D:\Resume classification Project\Resumes"

# Loop through folders
for root, dirs, files in os.walk(folder_path):

    for file in files:
        if file.endswith(".docx"):
            file_path = os.path.join(root, file)

            if root == folder_path:
                continue

            try:
                text = extract_text_from_docx(file_path)
                category = os.path.basename(root)

                data.append([text, category])

            except Exception as e:
                print(" Error reading:", file_path)

# Create DataFrame
df = pd.DataFrame(data, columns=["resume", "category"])

# ================== EDA ==================

print("\n📌 Shape of Data (Rows, Columns):")
print(df.shape)

print("\n📊 Data Types:")
print(df.dtypes)

print("\n📂 Number of Categories:")
print(df['category'].nunique())

print("\n📊 Category Names:")
print(df['category'].unique())

print("\n📈 Category Distribution:")
print(df['category'].value_counts())

print("\n🔍 Null Values Count:")
print(df.isnull().sum())

print("\n📉 Null Values Percentage (%):")
print((df.isnull().sum()/len(df))*100)

print("\n📏 Resume Length (characters):")
df['resume_length'] = df['resume'].apply(len)
print(df[['resume_length', 'category']].head())

print("\n📊 Resume Length Stats:")
print(df['resume_length'].describe())

# ================== NULL VALUE CHECK ==================

print("\n Null Value Count:\n")
print(df.isnull().sum())

print("\n Null Value Percentage (%):\n")
null_percent = (df.isnull().sum() / len(df)) * 100
print(null_percent)


# ====================== FEATURE ENGINEERING =====================

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming df is already created from EDA step

# 1️⃣ Resume length (characters)
df['resume_length'] = df['resume'].apply(len)

# 2️⃣ Word count
df['word_count'] = df['resume'].apply(lambda x: len(x.split()))

# 3️⃣ Category encoding (for model)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# 4️⃣ Number of unique words (vocabulary size)
df['unique_words'] = df['resume'].apply(lambda x: len(set(x.split())))

# 5️⃣ Average word length
df['avg_word_length'] = df['resume'].apply(lambda x: sum(len(w) for w in x.split()) / len(x.split()) if len(x.split())>0 else 0)

# 6️⃣ Count of uppercase words (can indicate emphasis / names)
df['uppercase_words'] = df['resume'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))

# ✅ Feature check
print("\nSample Feature Engineering Output:\n")
print(df[['resume_length','word_count','unique_words','avg_word_length','uppercase_words','category_encoded']].head())

