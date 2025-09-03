from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

movies = pd.DataFrame({
    'title': ['Inception', 'Interstellar', 'Dark Knight', 'Avengers'],
    'genre': ['Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Crime', 'Action Fantasy']
})

# TF-IDF feature extraction
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genre'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommend function
def recommend(title):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:3]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

print(recommend('Inception'))
