import pandas as pd

def load_movie_data(file_path):
    movies_df = pd.read_csv(file_path)
    return movies_df

def filter_relevant_data(movies_df):
    relevant_columns = ['title', 'genres', 'vote_average', 'vote_count']
    filtered_df = movies_df[relevant_columns]
    return filtered_df
