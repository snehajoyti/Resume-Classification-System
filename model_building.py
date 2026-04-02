# ==========================================DATA CLEANING FOR MODEL===============================================================

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import df

# Clean text: remove special characters, extra spaces, lowercase
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)          # multiple spaces to one
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text) # remove special chars
    text = text.lower()                        # lowercase
    return text.strip()

df['resume_clean'] = df['resume'].apply(clean_text)

#Text Vectorization
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # 5000 top words
X = tfidf.fit_transform(df['resume_clean'])

# Target
y = df['category_encoded']

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#======================================================MODEL BUILDING================================================================
# Initialize model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

#========================================================MODEL EVLAUTION===========================================================
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df['category'].unique(), yticklabels=df['category'].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import pickle

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model & Vectorizer Saved!")