import pandas as pd

def load_movie_data(file_path):
    movies_df = pd.read_csv(file_path)
    return movies_df
