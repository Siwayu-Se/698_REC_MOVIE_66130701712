
import streamlit as st
import pickle
import pandas as pd

# กำหนด URL หรือเส้นทางของภาพพื้นหลัง
## background_image_url = "https://images7.alphacoders.com/558/thumb-1920-558894.jpg"
background_image_url = "https://images6.alphacoders.com/133/thumbbig-1338130.webp"

# กำหนดสีที่ต้องการ
text_color = "#FFFFFF"  # สีที่คุณต้องการ

# ใส่ CSS สำหรับพื้นหลังและสีตัวอักษร
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        height: 100vh;
    }}
    h1, h2, h3, p, div {{
        color: {text_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model, movie ratings, and movie data
with open('recommendation_movie_svd_66130701712.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)


def get_top_recommendations(user_id, top_n=10):
    # Get movies rated by the user
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values

    # Filter out movies the user has already rated
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

    # Make predictions for unrated movies
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)

    # Get top movie recommendations
    top_recommendations = sorted_predictions[:top_n]

    # Get movie titles and additional information for the recommendations
    recommendations = [
        (
            movies[movies['movieId'] == recommendation.iid]['title'].values[0],
            movies[movies['movieId'] == recommendation.iid]['genres'].values[0],
            recommendation.est
        )
        for recommendation in top_recommendations
    ]

    return recommendations

# Streamlit UI setup
st.title("🎬 Movie Recommendation System By Siwayu")
st.write("Recommend top-rated movies for a specified user based on past ratings.")

# Sidebar for user input
st.sidebar.header("User Input")
user_id = st.sidebar.selectbox("Select User ID:", sorted(movie_ratings['userId'].unique()))
top_n = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=10)

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = get_top_recommendations(user_id, top_n=top_n)

        if recommendations:
            st.subheader(f"Top {top_n} Movie Recommendations for User {user_id}:")
            for idx, (title, genre, rating) in enumerate(recommendations, start=1):
                st.write(f"{idx}. **{title}** (Genre: {genre}) — Estimated Rating: {rating:.2f}")
        else:
            st.write("No recommendations available for this user.")
else:
    st.write("Select a user ID and click 'Get Recommendations' to see the suggestions.")

