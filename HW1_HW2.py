import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus_path = r"C:\Users\cyase\OneDrive\Documents\Corpus"
file_names = ["Baseball1.txt", "Baseball2.txt", "Baseball3.txt", "Videogames1.txt", "Videogames2.txt", "Videogames3.txt"]

labels = ["Baseball", "Baseball", "Baseball", "Videogames", "Videogames", "Videogames"]

documents = [open(os.path.join(corpus_path, file_name), 'r', encoding='utf-8').read() for file_name in file_names]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=[os.path.splitext(file)[0] for file in file_names])

df.insert(0, 'LABEL', labels)

output_csv_path = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\corpus_data_labeled.csv"
df.to_csv(output_csv_path, index=True)