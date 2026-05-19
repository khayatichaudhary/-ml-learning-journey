import pandas as pd
import ast
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[["title", "genres", "keywords", 
                 "overview", "vote_average"]]
movies["overview"] = movies["overview"].fillna("")

def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return " ".join([i["name"] for i in items])
    except:
        return ""

movies["genres"] = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)
movies["tags"] = movies["overview"] + " " + movies["genres"] + " " + movies["keywords"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["tags"])
similarity = cosine_similarity(tfidf_matrix)

# App UI
st.set_page_config(page_title="🎬 Movie Recommender", 
                   layout="wide")

st.title("🎬 Movie Recommender System")
st.markdown("### Find your next favorite movie!")
st.divider()

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    movie_name = st.selectbox("🎥 Select a movie:", 
                               movies["title"].values)
    
    recommend_btn = st.button("🎯 Recommend Movies!")
    surprise_btn = st.button("🎲 Surprise Me!")

# Surprise Me button
if surprise_btn:
    movie_name = random.choice(movies["title"].values)
    st.info(f"🎲 Random movie selected: **{movie_name}**")

# Recommend function
def show_recommendations(movie_name):
    try:
        index = movies[movies["title"] == movie_name].index[0]
        distances = sorted(list(enumerate(similarity[index])),
                          reverse=True, key=lambda x: x[1])
        
        st.divider()
        st.subheader(f"🎬 Movies similar to **{movie_name}**:")
        
        for i in distances[1:6]:
            movie = movies.iloc[i[0]]
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.write(f"🎬 **{movie.title}**")
                st.caption(f"🏷️ Genres: {movie.genres}")
            with col_b:
                st.metric("⭐ Rating", 
                          f"{movie.vote_average}/10")
            st.divider()
    except:
        st.error("Movie not found! Try another one!")

if recommend_btn:
    show_recommendations(movie_name)

if surprise_btn:
    show_recommendations(movie_name)