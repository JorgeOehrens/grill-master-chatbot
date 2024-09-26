import streamlit as st
import pandas as pd
from src.data_processing import load_movie_data
import matplotlib.pyplot as plt  
import time
from src.recommendation_engine import create_embeddings, movies_df_dim, top_five_recommendations, get_movie_title, representation_df
import faiss

page = st.sidebar.selectbox('Seleccionar Página', ['Inicio', 'Sistema de Recomendación', 'Sistema de Recomendación2', 'Documentación'])

# Definición de usuarios
user = pd.DataFrame([
    {
        'userId': 1,
        'username': 'John Doe',
        'email': 'john.doe@example.com',
        'password': 'password123',
        'profile_picture': 'images/user_2.png',
        'is_admin': False
    },
    {
        'userId': 2,
        'username': 'Jane Doe',
        'email': 'jane.doe@example.com',
        'password': 'password456',
        'profile_picture': 'images/user.png',
        'is_admin': False
    },
    {
        'userId': 3,
        'username': 'George',
        'email': 'admin@example.com',
        'password': 'admin123',
        'profile_picture': 'images/user_3.png',
        'is_admin': True
    }
])

# Definición de plataformas de streaming
stream_app = pd.DataFrame([
    {
        'stream_id': 1,
        'stream_name': 'Netflix',
        'stream_url': 'https://www.netflix.com',
        'stream_logo': 'https://www.netflix.com/favicon.ico',
        'stream_description': 'Netflix es una plataforma de streaming que ofrece una amplia variedad de películas y series.'
    },
    {
        'stream_id': 2,
        'stream_name': 'Amazon Prime Video',
        'stream_url': 'https://www.amazon.com/prime-video',
        'stream_logo': 'https://www.amazon.com/favicon.ico',
        'stream_description': 'Amazon Prime Video es una plataforma de streaming que ofrece una amplia variedad de películas y series.'
    },
    {
        'stream_id': 3,
        'stream_name': 'Hulu',
        'stream_url': 'https://www.hulu.com',
        'stream_logo': 'https://www.hulu.com/favicon.ico',
        'stream_description': 'Hulu es una plataforma de streaming que ofrece una amplia variedad de películas y series.'
    }
])

if page == 'Inicio':
    st.title('Bienvenido al Sistema de Recomendación de Películas')
    st.write('''
    Este es un sistema de recomendación de películas que utiliza técnicas de aprendizaje automático para sugerir películas a los usuarios.
    Aquí podrás encontrar películas que te interesarán según tus preferencias.
    ''')

elif page == 'Sistema de Recomendación':
    file_path = 'data/netflix_titles.csv'
    st.title('Sistema de Recomendación de Películas')
    movies_df = load_movie_data(file_path)

    selected_stream = st.selectbox('Selecciona una plataforma de streaming:', stream_app['stream_name'])
    st.image(stream_app[stream_app['stream_name'] == selected_stream]['stream_logo'].values[0], width=50)

    movie_input = st.text_input('Introduce una película para obtener recomendaciones:')
    director_favorite = st.text_input('Introduce tu director favorito:')
    cast_favorite = st.text_input('Introduce tu actor/actriz favorito:')
    genre_favorite = st.text_input('Introduce tu género favorito:')

    if st.button('Buscar'):
        favorite_movie = {
            'title': movie_input,
            'director': director_favorite,
            'cast': cast_favorite,
            'listed_in': genre_favorite,
            'description': movie_input + " " + director_favorite + " " + cast_favorite + " " + genre_favorite  # Asegúrate de que este campo se maneje correctamente si es relevante
        }
        
        representation = representation_df(favorite_movie)
        embedding = create_embeddings(representation)

        I = top_five_recommendations(embedding)
        movies_df = movies_df_dim()
        movies_df['textual_representation'] = movies_df.apply(representation_df, axis=1)

        best_matches = get_movie_title(I, movies_df)
    
        for i, match in enumerate(best_matches, 1):
            st.write(f"{i}. {match}")
    
    
    

elif page == 'Sistema de Recomendación2':
    file_path = 'data/tmdb_5000_movies.csv'
    st.title('Selecciona la siguiene pelicula que verá el personaje ')
    movies_df = load_movie_data(file_path)
    filtered_movies_df = filter_relevant_data(movies_df)

    selected_user = st.selectbox('Selecciona un usuario:', user['username'])
    st.image(user[user['username'] == selected_user]['profile_picture'].values[0], width=50)
    selected_stream = st.selectbox('Selecciona una plataforma de streaming:', stream_app['stream_name'])
    st.image(stream_app[stream_app['stream_name'] == selected_stream]['stream_logo'].values[0], width=50)


    
    
elif page == 'Documentación':
    st.title('Documentación del Sistema de Recomendación de Películas')
    st.write('''
    Este sistema de recomendación de películas utiliza técnicas de aprendizaje automático para sugerir películas a los usuarios.
    Aquí se describe el proceso y los componentes clave del sistema.
    ## 1. Preparación de Datos
    - **Carga de Datos**: Se cargan los datos de películas desde un archivo CSV.
    - **Filtrado de Datos Relevantes**: Se filtran los datos para incluir solo las películas relevantes.
    - **Creación de Matriz de Películas del Usuario**: Se crea una matriz que representa las películas que los usuarios han visto.
    ## 2. Cálculo de Similitudes
    - **Cálculo de Similitudes**: Se calcula la similitud entre las películas basándose en sus características.
    ## 3. Recomendación de Películas
    - **Recomendación de Películas**: Se utilizan las similitudes calculadas para sugerir películas a los usuarios.
    ## 4. Visualización de Datos
    - **Visualización de Datos**: Se utilizan gráficos de pandas para mostrar los datos de películas y los perfiles de usuario.
    ''')
    st.subheader('Visualización de Películas')
    fig, ax = plt.subplots()
    movies_df['title'].value_counts().head(20).plot(kind='barh', ax=ax)
    ax.set_xlabel('Número de Apariciones')
    ax.set_ylabel('Películas')
    st.pyplot(fig)
    st.write('''
    ## 5. Interfaz de Usuario
    - **Interfaz de Usuario**: Se crea una interfaz de usuario simple para que los usuarios puedan seleccionar películas y recibir recomendaciones.
    ''')
    st.subheader('Visualización de Perfiles de Usuario')
    fig, ax = plt.subplots()
    user['username'].value_counts().plot(kind='barh', ax=ax)
    ax.set_xlabel('Número de Usuarios')
    ax.set_ylabel('Perfiles de Usuario')
    st.pyplot(fig)