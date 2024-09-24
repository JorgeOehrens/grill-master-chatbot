import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data():
    return pd.read_csv('data/tmdb_5000_movies.csv')

def create_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'].fillna(''))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, cosine_sim, indices, data):
    if title not in indices:
        return pd.Series()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

def get_movie_recommendations(user_preferences):
    data = load_data()
    cosine_sim = create_similarity_matrix(data)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    
    recommendations = get_recommendations(user_preferences, cosine_sim, indices, data)
    if not recommendations.empty:
        response = "Based on your preferences, we recommend: " + ", ".join(recommendations)
    else:
        response = "Sorry, no movies found matching your preferences."
    return response