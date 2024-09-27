from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import faiss

load_dotenv()

def representation_df(row):
    textual_representation = f"Title: {row['title']}, Director: {row['director']}, Cast: {row['cast']},  Genres: {row['listed_in']}, Description: {row['description']}"
    return textual_representation

def represent_embeddings(movies_df):
    movies_df['textual_representation'] = movies_df.apply(representation_df, axis=1)
    return movies_df

def create_embeddings(favorite_movie):
    print(favorite_movie)
    query_vector = OpenAIEmbeddings().embed_query(favorite_movie)
    return query_vector

def top_five_recommendations(query_vector):
    index = faiss.read_index("src/embedding/index2.faiss")
    D, I = index.search(np.array([query_vector]), k=5)
    print("Distancias:", D)  
    print("Índices:", I)   
    return I

def get_movie_title(I, movies_df):
    best_matches = np.array(movies_df['title'])[I.flatten()]
    print("Mejores coincidencias:", best_matches) 
    return best_matches

def movies_df_dim():
    movies_df = pd.read_csv("data/netflix_titles.csv")
    return movies_df