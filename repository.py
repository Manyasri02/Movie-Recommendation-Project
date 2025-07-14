#Import libraries and load data
import pandas as pd

# Load the movies metadata
movies = pd.read_csv('tmdb_5000_movies.csv')

print(movies.head())
#Preprocess data (select features, clean data)
import ast

# We'll use 'title', 'genres', 'keywords', 'overview'
def parse_features(x):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(x)])
    except:
        return ""

movies['genres'] = movies['genres'].apply(parse_features)
movies['keywords'] = movies['keywords'].apply(parse_features)
movies['overview'] = movies['overview'].fillna('')

def combine_features(row):
    return f"{row['genres']} {row['keywords']} {row['overview']}"

movies['combined_features'] = movies.apply(combine_features, axis=1)

print(movies[['title', 'combined_features']].head())

#Convert text features to numerical vectors (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

#Compute similarity scores (Cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies based on similarity
# Reset index for easy lookup
movies = movies.reset_index()

def recommend_movies(title, cosine_sim=cosine_sim, movies=movies):
    # Find the index of the movie that matches the title
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return ["Movie not found."]
    idx = idx[0]
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top 5, ignoring the first (itself)
    movie_indices = [i[0] for i in sim_scores[1:6]]
    return movies['title'].iloc[movie_indices].tolist()

# Example usage
print(recommend_movies('Avatar'))

user_title = input("Enter a movie title: ")
print("Recommended movies:")
for movie in recommend_movies(user_title):
    print(movie)