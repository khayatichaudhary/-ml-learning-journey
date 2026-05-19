# import pandas as pd

# # Load the data
# movies = pd.read_csv("tmdb_5000_movies.csv")
# print(movies.head())
# print(movies.shape)
# print(movies.columns)

# # Select only useful columns
# movies = movies[["title", "genres", "keywords", "overview"]]

# # Check for missing values
# print(movies.isnull().sum())

# # See first row
# print(movies.head(1))
# # Fill missing overview with empty string
# movies["overview"] = movies["overview"].fillna("")

# # Create tags by combining all columns
# movies["tags"] = movies["overview"] + " " + movies["genres"] + " " + movies["keywords"]

# # result
# print(movies[["title", "tags"]].head(2))
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Convert tags to numbers
# tfidf = TfidfVectorizer(stop_words="english")
# tfidf_matrix = tfidf.fit_transform(movies["tags"])

# # Calculate similarity between all movies
# similarity = cosine_similarity(tfidf_matrix)

# print("Similarity matrix shape:", similarity.shape)

# def recommend(movie_name):
#     # Find movie index
#     index = movies[movies["title"] == movie_name].index[0]
    
#     # Get similarity scores for this movie
#     distances = sorted(list(enumerate(similarity[index])), 
#                       reverse=True, key=lambda x: x[1])
    
#     # Print top 5 similar movies
#     print(f"\nMovies similar to {movie_name}:")
#     for i in distances[1:6]:
#         print(movies.iloc[i[0]].title)

# # Test it!
# recommend("Inception")
# print(movies["genres"][0])
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")

# Select useful columns
movies = movies[["title", "genres", "keywords", "overview"]]

# Fix missing values
movies["overview"] = movies["overview"].fillna("")

# Function to extract names from JSON
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return " ".join([i["name"] for i in items])
    except:
        return ""

# Clean genres and keywords
movies["genres"] = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)

# Create tags
movies["tags"] = movies["overview"] + " " + movies["genres"] + " " + movies["keywords"]

# Convert tags to numbers
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["tags"])

# Calculate similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommender function
def recommend(movie_name):
    index = movies[movies["title"] == movie_name].index[0]
    distances = sorted(list(enumerate(similarity[index])),
                      reverse=True, key=lambda x: x[1])
    print(f"\nMovies similar to {movie_name}:")
    for i in distances[1:6]:
        print(movies.iloc[i[0]].title)

# Test!
recommend("Avatar")
recommend("Inception")
recommend("The Dark Knight")