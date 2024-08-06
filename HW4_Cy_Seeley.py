import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "C:/Users/cyase/OneDrive/Documents/Syracuse Masters/Text Mining/deception_data_two_labels.csv"
data = pd.read_csv(file_path)

data['full_review'] = data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
data = data[['lie', 'sentiment', 'full_review']]

data['full_review'] = data['full_review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower())

stop_words = set(stopwords.words('english'))
data['full_review'] = data['full_review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

data.dropna(inplace=True)

cleaned_file_path = "C:/Users/cyase/OneDrive/Documents/Syracuse Masters/Text Mining/cleaned_deception_data.xlsx"
data.to_excel(cleaned_file_path, index=False)

cleaned_data = pd.read_excel(cleaned_file_path)

sentiment_data = cleaned_data[['full_review', 'sentiment']]
lie_detection_data = cleaned_data[['full_review', 'lie']]

X_sentiment = sentiment_data['full_review']
y_sentiment = sentiment_data['sentiment']
X_lie_detection = lie_detection_data['full_review']
y_lie_detection = lie_detection_data['lie']

def train_and_evaluate_model(X, y, label_name):
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    
    scores = cross_val_score(model, X_vectorized, y, cv=10)
    print(f"10-Fold Cross-Validation Accuracy for {label_name}: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
    
    feature_names = vectorizer.get_feature_names_out()
    most_indicative = np.argsort(model.feature_log_prob_[1])[-20:]
    indicative_words = [feature_names[i] for i in most_indicative]
    print(f"20 Most Indicative Words for {label_name}: {indicative_words}")
    
    return indicative_words

indicative_words_sentiment = train_and_evaluate_model(X_sentiment, y_sentiment, "Sentiment Classification")
indicative_words_lie_detection = train_and_evaluate_model(X_lie_detection, y_lie_detection, "Lie Detection")


models = ['Sentiment Classification', 'Lie Detection']
accuracies = [0.84, 0.62]  # Mean accuracies
errors = [0.14, 0.10]  # Standard deviations


data = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'Error': errors
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=data, ci=None, palette='muted')

plt.errorbar(x=data['Model'], y=data['Accuracy'], yerr=data['Error'], fmt='o', color='black', capsize=5)

plt.title('Accuracy Comparison of Sentiment Classification and Lie Detection Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.show()
