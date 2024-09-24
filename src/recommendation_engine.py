from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_user_movie_matrix(movies_df, user_ratings_df):
    user_movie_matrix = pd.pivot_table(user_ratings_df, index='userId', columns='movieId', values='rating')
    return user_movie_matrix.fillna(0)

def calculate_similarity(user_movie_matrix):
    similarity_matrix = cosine_similarity(user_movie_matrix.T)
    return similarity_matrix

def recommend_movies(movie_id, similarity_matrix, movies_df, top_n=5):
    movie_index = movies_df[movies_df['id'] == movie_id].index[0]
    similar_movies = sorted(list(enumerate(similarity_matrix[movie_index])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_movies = [movies_df.iloc[i[0]]['title'] for i in similar_movies]
    return recommended_movies
