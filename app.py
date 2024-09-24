import streamlit as st
import pandas as pd
from src.data_processing import load_movie_data, filter_relevant_data
from src.recommendation_engine import recommend_movies, create_user_movie_matrix, calculate_similarity
from src.visualization import display_movie_data, display_movie_recommendations

file_path = 'data/tmdb_5000_movies.csv'

st.title('Sistema de Recomendación de Películas')

movies_df = load_movie_data(file_path)
filtered_movies_df = filter_relevant_data(movies_df)
display_movie_data(filtered_movies_df)

movie_input = st.text_input('Introduce una película para obtener recomendaciones:')

user_ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])  
if movie_input:
    movie_matches = movies_df[movies_df['title'].str.contains(movie_input, case=False)]

    if not movie_matches.empty:
        selected_movie = st.selectbox('Selecciona una película:', movie_matches['title'])

        movie_id = movie_matches[movie_matches['title'] == selected_movie]['id'].values[0]

        # Corrección: usar pd.concat para añadir una nueva fila al DataFrame
        user_ratings_df = pd.concat([user_ratings_df, pd.DataFrame([{
            'userId': 1,  
            'movieId': movie_id, 
        }])])

        user_movie_matrix = create_user_movie_matrix(movies_df, user_ratings_df)
        similarity_matrix = calculate_similarity(user_movie_matrix)

        recommended_movies = recommend_movies(movie_id, similarity_matrix, movies_df)

        display_movie_recommendations(recommended_movies)
    else:
        st.warning('No se encontraron películas que coincidan con tu búsqueda.')