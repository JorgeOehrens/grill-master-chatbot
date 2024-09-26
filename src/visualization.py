import streamlit as st
import pandas as pd

def display_movie_recommendations(recommended_movies):
    st.write("### Películas Recomendadas:")
    for movie in recommended_movies:
        st.write(f"- {movie}")

def display_movie_data(movies_df):
    st.write("### Datos de Películas:")
    st.dataframe(movies_df)
