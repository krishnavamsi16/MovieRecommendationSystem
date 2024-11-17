import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import GridSearchCV
from collections import defaultdict
from surprise import KNNBasic, SVD, SVDpp, Dataset, Reader
from surprise.model_selection import cross_validate
# Load and preprocess data
@st.cache_data
def load_data():
    ratings = pd.read_csv('E:/Internship/IITROORKEE-DataScience,ML,AIinPython/MovieRecommFinalProject/ml-32m/ratings1.csv')
    movies = pd.read_csv('E:/Internship/IITROORKEE-DataScience,ML,AIinPython/MovieRecommFinalProject/ml-32m/movies.csv')
    links = pd.read_csv('E:/Internship/IITROORKEE-DataScience,ML,AIinPython/MovieRecommFinalProject/ml-32m/links.csv')
    tags = pd.read_csv('E:/Internship/IITROORKEE-DataScience,ML,AIinPython/MovieRecommFinalProject/ml-32m/tags.csv')
    
    # Merge movies with links
    movies = pd.merge(movies, links[['movieId', 'tmdbId']], on='movieId', how='left')
    
    # Aggregate tags for each movie, handling non-string tags
    def aggregate_tags(group):
        # Convert all tags to strings and filter out NaN values
        valid_tags = [str(tag) for tag in group if pd.notna(tag)]
        return ', '.join(set(valid_tags)) if valid_tags else "No tags available"
    
    tags_agg = tags.groupby('movieId')['tag'].agg(aggregate_tags).reset_index()
    movies = pd.merge(movies, tags_agg, on='movieId', how='left')
    
    return ratings, movies

ratings, movies = load_data()

# Data Exploration and Preprocessing
def preprocess_data(ratings, movies):
    st.write("Data Exploration")
    
    st.write("Ratings Overview")
    st.write(ratings.describe())
    
    st.write("Movies Overview")
    st.write(movies.describe())
    
    st.write("Ratings Distribution")
    fig, ax = plt.subplots()
    sns.histplot(ratings['rating'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write("Top 10 Most Rated Movies")
    top_movies = ratings['movieId'].value_counts().head(10)
    top_movie_titles = movies[movies['movieId'].isin(top_movies.index)]['title']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_movies.values, y=top_movie_titles, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    return ratings, movies

ratings, movies = preprocess_data(ratings, movies)

def fetch_poster(tmdb_id):
    api_key = '851f9064810df850e4eb9bff434ff962'  # Replace with your actual API key
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")
    
    return "https://via.placeholder.com/500x750?text=No+Poster+Available"


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

def get_user_based_recommendations(user_id, model, movies, top_n=5):
    # Get all movie IDs
    all_movie_ids = movies['movieId'].unique()
    
    # Predict ratings for all movies
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movie_ids]
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top N recommendations
    top_recs = predictions[:top_n]
    
    # Get movie details for recommendations
    recommended_movies = []
    for pred in top_recs:
        movie_id = pred.iid
        movie_info = movies[movies['movieId'] == movie_id].iloc[0]
        recommended_movies.append({
            'title': movie_info['title'],
            'movieId': movie_id,
            'tmdbId': movie_info['tmdbId'],
            'tags': movie_info['tag'] if pd.notna(movie_info['tag']) else "No tags available"
        })
    
    return recommended_movies

def get_item_based_recommendations(movie_name, model, movies, ratings, top_n=5):
    movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
    # Get users who rated this movie
    users_who_rated = ratings[ratings['movieId'] == movie_id]['userId'].unique()
    # Get all other movies
    other_movies = movies[movies['movieId'] != movie_id]['movieId'].unique()
    # Predict ratings for other movies
    predictions = []
    for user_id in users_who_rated:
        user_predictions = [model.predict(user_id, other_movie_id) for other_movie_id in other_movies]
        predictions.extend(user_predictions)
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    # Get top N unique recommendations
    top_recs = []
    seen_movies = set()
    for pred in predictions:
        if pred.iid not in seen_movies:
            top_recs.append(pred)
            seen_movies.add(pred.iid)
        if len(top_recs) == top_n:
            break
    # Get movie details for recommendations
    recommended_movies = []
    for pred in top_recs:
        movie_id = pred.iid
        movie_info = movies[movies['movieId'] == movie_id].iloc[0]
        recommended_movies.append({
            'title': movie_info['title'],
            'movieId': movie_id,
            'tmdbId': movie_info['tmdbId'],
            'tags': movie_info['tag'] if pd.notna(movie_info['tag']) else "No tags available"
        })
    
    return recommended_movies

@st.cache_resource
def train_and_evaluate_model(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Use a subset of the data for faster computation
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    
    # Define the parameter grid
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    
    # Perform grid search
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    
    # Get the best parameters
    best_params = gs.best_params['rmse']
    
    # Train the model with the best parameters
    best_model = SVD(**best_params)
    best_model.fit(trainset)
    
    # Make predictions on the test set
    predictions = best_model.test(testset)
    
    # Compute RMSE
    rmse = accuracy.rmse(predictions)
    
    return best_model, rmse

# Streamlit UI
st.title('MovieLens Recommendation System')

user_list = ratings['userId'].unique().tolist()
movie_list = movies['title'].tolist()

selected_user = st.selectbox('Select User ID', user_list)
selected_movie = st.selectbox('Choose a movie for  recommendation', movie_list)

# Train and evaluate model
if 'model' not in st.session_state:
    with st.spinner('Training and evaluating model... This may take a while.'):
        model, rmse = train_and_evaluate_model(ratings)
        st.session_state['model'] = model
        st.session_state['rmse'] = rmse
        #st.write(f"Model RMSE: {rmse:.4f}")

if st.button('Get User-Based Recommendations'):
    user_recommendations = get_user_based_recommendations(selected_user, st.session_state['model'], movies)
    if user_recommendations:
        st.write("User-based recommendations:")
        for movie in user_recommendations:
            st.write(f"Title: {movie['title']}")
            st.write(f"Tags: {movie['tags']}")
            poster_url = fetch_poster(movie['tmdbId'])
            if poster_url:
                st.image(poster_url, width=200)
            else:
                st.write("No poster available.")
    else:
        st.write("No recommendations available.")

if st.button('Get Item-Based Recommendations'):
    item_recommendations = get_item_based_recommendations(selected_movie, st.session_state['model'], movies, ratings)
    if item_recommendations:
        st.write("Item-based recommendations:")
        for movie in item_recommendations:
            st.write(f"Title: {movie['title']}")
            st.write(f"Tags: {movie['tags']}")
            poster_url = fetch_poster(movie['tmdbId'])
            if poster_url:
                st.image(poster_url, width=200)
            else:
                st.write("No poster available.")
    else:
        st.write("No recommendations available.")

# Display RMSE
st.write(f"Model RMSE: {st.session_state['rmse']:.4f}")



# Define a reader to parse the rating data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# User-based Collaborative Filtering
user_cf = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
user_cf_results = cross_validate(user_cf, data, measures=['RMSE'], cv=5, verbose=True)
print("User-based CF RMSE: ", user_cf_results['test_rmse'].mean())
# Item-based Collaborative Filtering
item_cf = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
item_cf_results = cross_validate(item_cf, data, measures=['RMSE'], cv=5, verbose=True)
print("Item-based CF RMSE: ", item_cf_results['test_rmse'].mean())
# SVD Model (Matrix Factorization)
svd = SVD()
svd_results = cross_validate(svd, data, measures=['RMSE'], cv=5, verbose=True)
print("SVD RMSE: ", svd_results['test_rmse'].mean())
# SVD++ Model (Matrix Factorization with implicit feedback)
svdpp = SVDpp()
svdpp_results = cross_validate(svdpp, data, measures=['RMSE'], cv=5, verbose=True)
print("SVD++ RMSE: ", svdpp_results['test_rmse'].mean())
print("User-based CF RMSE: ", user_cf_results['test_rmse'].mean())
print("Item-based CF RMSE: ", item_cf_results['test_rmse'].mean())
print("SVD RMSE: ", svd_results['test_rmse'].mean())
print("SVD++ RMSE: ", svdpp_results['test_rmse'].mean())

