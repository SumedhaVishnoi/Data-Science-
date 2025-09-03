import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item rating matrix
data = {
    'Movie1': [5, 4, 0, 0, 1],
    'Movie2': [4, 0, 0, 2, 1],
    'Movie3': [1, 1, 0, 5, 4],
    'Movie4': [0, 0, 5, 4, 0]
}

ratings = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4', 'User5'])
print("User-Item Ratings:\n", ratings)

# Step 1: Compute similarity between users
user_similarity = cosine_similarity(ratings)
user_sim_df = pd.DataFrame(user_similarity, index=ratings.index, columns=ratings.index)

print("\nUser Similarity:\n", user_sim_df)

# Step 2: Recommend based on similar users
def recommend(user, n=2):
    similar_users = user_sim_df[user].sort_values(ascending=False)[1:n+1].index
    recommendations = ratings.loc[similar_users].mean().sort_values(ascending=False)
    return recommendations

print("\nRecommendations for User1:\n", recommend('User1'))
