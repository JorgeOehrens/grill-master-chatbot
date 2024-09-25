from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_user_movie_matrix(movies_df, user_ratings_df):
    required_columns = {'userId', 'movieId', 'rating'}
    if not required_columns.issubset(user_ratings_df.columns):
        raise ValueError(f"Faltan columnas requeridas en user_ratings_df. Columnas requeridas: {required_columns}")

    user_ratings_df['userId'] = user_ratings_df['userId'].astype(int)
    user_ratings_df['movieId'] = user_ratings_df['movieId'].astype(int)
    user_ratings_df['rating'] = user_ratings_df['rating'].astype(float)

    user_movie_matrix = pd.pivot_table(user_ratings_df, index='userId', columns='movieId', values='rating')
    return user_movie_matrix.fillna(0)
def calculate_similarity(user_movie_matrix):
    if user_movie_matrix.shape[0] < 2:
        raise ValueError("Insufficient data to calculate similarity.")
    return cosine_similarity(user_movie_matrix)

def recommend_movies(movie_id, similarity_matrix, movies_df):
    print(f"Debug: movie_id = {movie_id}")
    movie_indices = pd.Series(range(len(movies_df)), index=movies_df['id'])
    
    if movie_id not in movie_indices:
        raise ValueError("Movie ID not found in the dataset.")
    
    movie_index = movie_indices[movie_id]
    print(f"Debug: movie_index = {movie_index}, similarity_matrix size = {similarity_matrix.shape}")
    
    if movie_index >= similarity_matrix.shape[0]:
        raise IndexError(f"Index {movie_index} is out of bounds for similarity matrix with size {similarity_matrix.shape[0]}")
    
    similar_movies = sorted(list(enumerate(similarity_matrix[movie_index])), key=lambda x: x[1], reverse=True)[1:11]
    recommended_movie_ids = [movies_df.iloc[i[0]]['id'] for i in similar_movies]
    recommended_movies = movies_df[movies_df['id'].isin(recommended_movie_ids)]
    
    return recommended_movies['title'].tolist()