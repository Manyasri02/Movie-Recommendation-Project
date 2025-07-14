import streamlit as st
from repository import recommend_movies, movies



st.title("Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend_movies(selected_movie)
    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)