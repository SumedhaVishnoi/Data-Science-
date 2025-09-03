import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------
# Step 1: Movie dataset
# ----------------------------
movies = pd.DataFrame({
    'movieId': [1, 2, 3, 4, 5],
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Iron Man'],
    'genre': ['Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Crime', 'Action Fantasy', 'Action Sci-Fi']
})

ratings = pd.DataFrame({
    'userId': [1, 1, 2, 2, 3, 3, 4],
    'movieId': [1, 2, 2, 3, 3, 4, 5],
    'rating': [5, 4, 5, 4, 3, 5, 4]
})

# ----------------------------
# Step 2: Content-based similarity (TF-IDF on genres)
# ----------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ----------------------------
# Step 3: Collaborative filtering (user-item average ratings)
# ----------------------------
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
collab_sim = cosine_similarity(user_item_matrix.T)  # item-item similarity
collab_sim_df = pd.DataFrame(collab_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# ----------------------------
# Step 4: Hybrid Recommendation
# ----------------------------
def hybrid_recommend(movie_title, top_n=3):
    # Find movie index
    idx = movies[movies['title'] == movie_title].index[0]
    movie_id = movies.loc[idx, 'movieId']
    
    # Content similarity scores
    content_scores = list(enumerate(content_sim[idx]))
    
    # Collaborative similarity scores
    collab_scores = list(enumerate(collab_sim_df[movie_id]))
    
    # Combine (weighted average: 50% content + 50% collaborative)
    hybrid_scores = {}
    for i, c_score in content_scores:
        col_score = collab_scores[i][1] if i < len(collab_scores) else 0
        hybrid_scores[i] = 0.5 * c_score + 0.5 * col_score
    
    # Sort and recommend
    sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sorted_scores]
    return movies['title'].iloc[movie_indices]

# ----------------------------
# Example Run
# ----------------------------
print("Recommendations for 'Inception':")
print(hybrid_recommend('Inception'))
