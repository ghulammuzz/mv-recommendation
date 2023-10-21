import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

df = pd.read_csv("mov.csv")

df['description'] = df['description'].fillna('')  # Replace missing values with empty string

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

with open('recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(cosine_sim, model_file)
