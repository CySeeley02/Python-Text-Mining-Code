import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

file_path = 'C:/Users/cyase/OneDrive/Documents/Syracuse Masters/Text Mining/T_F_Reviews.csv'
df = pd.read_csv(file_path)

print(df.head())

df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

df.dropna(inplace=True)

df.rename(columns={'lie': 'LABEL', 'review': 'REVIEW'}, inplace=True)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['REVIEW'])

word_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

df = pd.concat([df.drop(columns=['REVIEW']), word_matrix], axis=1)

print(df.head())

cleaned_file_path = 'C:/Users/cyase/OneDrive/Documents/Syracuse Masters/T_F_Reviews_Cleaned.csv'
df.to_csv(cleaned_file_path, index=False)
