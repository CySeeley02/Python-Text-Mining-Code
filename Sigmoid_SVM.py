# svm_sigmoid.py

import os
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string

# File paths for positive and negative review zip files
pos_zip_path = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\pos-20240820T011722Z-001.zip"
neg_zip_path = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\neg-20240820T011834Z-001.zip"

# Function to load reviews from a zip file
def load_reviews_from_zip(zip_path, label):
    reviews = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for idx, filename in enumerate(zip_ref.namelist()):
            with zip_ref.open(filename) as file:
                reviews.append({'review': file.read().decode('utf-8'), 'sentiment': label})
    return reviews

# Load positive and negative reviews from zip files
pos_reviews = load_reviews_from_zip(pos_zip_path, 'pos')
neg_reviews = load_reviews_from_zip(neg_zip_path, 'neg')

# Combine the datasets into a single DataFrame
df = pd.DataFrame(pos_reviews + neg_reviews)

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Feature Extraction using CountVectorizer
vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].map({'pos': 1, 'neg': 0})

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standard Scaler for SVM models
scaler = StandardScaler(with_mean=False)

# Pipeline for Sigmoid kernel
pipe_sigmoid = Pipeline([
    ('scaler', scaler),
    ('svc', SVC(kernel='sigmoid', max_iter=1000))
])
param_grid_sigmoid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}
grid_sigmoid = GridSearchCV(pipe_sigmoid, param_grid_sigmoid, cv=3, verbose=1, n_jobs=-1)
grid_sigmoid.fit(X_train, y_train)
best_sigmoid_model = grid_sigmoid.best_estimator_
sigmoid_preds = best_sigmoid_model.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, sigmoid_preds)
print("Optimized SVM with Sigmoid Kernel Confusion Matrix:")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized SVM with Sigmoid Kernel Confusion Matrix')
plt.show()

print("Optimized SVM with Sigmoid Kernel Classification Report:")
print(classification_report(y_test, sigmoid_preds))

accuracy = accuracy_score(y_test, sigmoid_preds)
print(f"Optimized SVM with Sigmoid Kernel Accuracy: {accuracy:.4f}")
