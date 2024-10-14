import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import string
from nltk.corpus import stopwords
import re
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
filepath = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\deception_data_two_labels.csv"
df = pd.read_csv(filepath)

# Assuming 'lie' is the label you want to predict; you can also use 'sentiment'
labels = df['lie']

# Load stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation and lowercase the text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove words that are stopwords, have numbers, have length 2 or less, or are longer than 13 characters
    text = ' '.join([
        word for word in text.split() 
        if word not in stop_words 
        and len(word) > 2 
        and len(word) <= 13 
        and not re.search(r'\d', word)
    ])
    return text

# Apply preprocessing to the 'review' text data
df['clean_review'] = df['review'].astype(str).apply(preprocess_text)

# Save cleaned data to a CSV file
cleaned_filepath = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\cleaned_deception_data.csv"
df.to_csv(cleaned_filepath, index=False)

# Vectorization for Bernoulli Naive Bayes
vectorizer_bernoulli = CountVectorizer(binary=True)
X_bernoulli = vectorizer_bernoulli.fit_transform(df['clean_review'])

# Vectorization for Multinomial Naive Bayes and Decision Tree
vectorizer_multinomial = CountVectorizer(binary=False)
X_multinomial = vectorizer_multinomial.fit_transform(df['clean_review'])

# Split the data into training and testing sets
X_train_bern, X_test_bern, y_train, y_test = train_test_split(X_bernoulli, labels, test_size=0.2, random_state=42)
X_train_multi, X_test_multi, _, _ = train_test_split(X_multinomial, labels, test_size=0.2, random_state=42)

# Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train_bern, y_train)
y_pred_bern = bernoulli_nb.predict(X_test_bern)

# Confusion Matrix and Accuracy
cm_bern = confusion_matrix(y_test, y_pred_bern)
accuracy_bern = accuracy_score(y_test, y_pred_bern)

# Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_multi, y_train)
y_pred_multi = multinomial_nb.predict(X_test_multi)

# Confusion Matrix and Accuracy
cm_multi = confusion_matrix(y_test, y_pred_multi)
accuracy_multi = accuracy_score(y_test, y_pred_multi)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_multi, y_train)
y_pred_tree = decision_tree.predict(X_test_multi)

# Confusion Matrix and Accuracy
cm_tree = confusion_matrix(y_test, y_pred_tree)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Display confusion matrices and accuracies
print("Bernoulli Naive Bayes Confusion Matrix:\n", cm_bern)
print("Bernoulli Naive Bayes Accuracy: ", accuracy_bern)

print("\nMultinomial Naive Bayes Confusion Matrix:\n", cm_multi)
print("Multinomial Naive Bayes Accuracy: ", accuracy_multi)

print("\nDecision Tree Classifier Confusion Matrix:\n", cm_tree)
print("Decision Tree Classifier Accuracy: ", accuracy_tree)

# Plot Confusion Matrices with legends
plt.figure(figsize=(18, 6))

# Bernoulli Naive Bayes
plt.subplot(1, 3, 1)
sns.heatmap(cm_bern, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Bernoulli Naive Bayes")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.legend(["Truthful", "Deceptive"])

# Multinomial Naive Bayes
plt.subplot(1, 3, 2)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("Multinomial Naive Bayes")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.legend(["Truthful", "Deceptive"])

# Decision Tree Classifier
plt.subplot(1, 3, 3)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title("Decision Tree Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.legend(["Truthful", "Deceptive"])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Set up the plot with a larger figure size for Decision Tree visualization
plt.figure(figsize=(40, 20))

# Limit the depth of the tree to avoid clutter (you can adjust max_depth as needed)
plot_tree(decision_tree, 
          feature_names=vectorizer_multinomial.get_feature_names_out(), 
          class_names=['Truthful', 'Deceptive'], 
          filled=True, 
          rounded=True, 
          fontsize=10,   # Font size to keep text readable
          precision=2,   # Adjust precision for numeric values
          max_depth=3    # Limiting the depth for better clarity
         )

# Add a title to the plot
plt.title("Decision Tree Visualization for Deception Detection", fontsize=20)

# Adjust layout to avoid overlap
plt.tight_layout()

# Display the plot
plt.show()

# Visualization: Raw vs. Cleaned Data
plt.figure(figsize=(12, 6))

# Raw Data Lengths
plt.subplot(1, 2, 1)
df['raw_length'] = df['review'].apply(lambda x: len(str(x).split()))
plt.hist(df['raw_length'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Raw Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

# Cleaned Data Lengths
plt.subplot(1, 2, 2)
df['clean_length'] = df['clean_review'].apply(lambda x: len(str(x).split()))
plt.hist(df['clean_length'], bins=20, color='green', alpha=0.7)
plt.title('Distribution of Cleaned Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Feature Importance for Decision Tree
feature_importances_tree = pd.DataFrame({
    'Feature': vectorizer_multinomial.get_feature_names_out(),
    'Importance': decision_tree.feature_importances_
})
feature_importances_tree = feature_importances_tree.sort_values(by='Importance', ascending=False).head(20)

# Plot the top 20 most important features for Decision Tree
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_tree)
plt.title("Top 20 Feature Importance for Decision Tree")
plt.show()

# Error Analysis for Multinomial Naive Bayes
df_test_multi = df.iloc[y_test.index].copy()  # Get the original test data
df_test_multi['Predicted'] = y_pred_multi  # Add predicted labels
df_test_multi['Actual'] = y_test  # Add actual labels
df_test_multi['Correct'] = df_test_multi['Predicted'] == df_test_multi['Actual']  # Add correctness

# Analyze errors
errors_multi = df_test_multi[df_test_multi['Correct'] == False]
print("\nMultinomial Naive Bayes Error Analysis:")
print(errors_multi[['review', 'Actual', 'Predicted']].head(10))  # Display first 10 errors


# Bar Plot of Error Counts
error_counts = errors_multi.groupby(['Actual', 'Predicted']).size().unstack(fill_value=0)
error_counts.plot(kind='bar', stacked=True, colormap='Reds', figsize=(10, 6))
plt.title("Error Counts by Actual vs. Predicted Labels")
plt.xlabel("Actual Labels")
plt.ylabel("Number of Errors")
plt.show()

# Distribution of Predicted Probabilities for Bernoulli Naive Bayes
y_prob_bern = bernoulli_nb.predict_proba(X_test_bern)[:, 1]

plt.figure(figsize=(10, 6))
sns.histplot(y_prob_bern, bins=20, kde=True)
plt.title("Distribution of Predicted Probabilities for Bernoulli Naive Bayes")
plt.xlabel("Predicted Probability of Being Deceptive")
plt.ylabel("Frequency")
plt.show()




