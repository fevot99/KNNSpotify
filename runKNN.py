# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name and get 5 similar song recommendations based on KNN.")

# Load your preprocessed dataset (assuming you have a dataframe `df` with 'song', 'artist', and feature columns)
# Example DataFrame structure:
df = pd.read_csv('testpca.csv') # Preprocessed music data with numerical features

# Load the trained KNN model from the pickle file
with open('knn_model.pkl', 'rb') as f:
    knn10 = pickle.load(f)

# Use the relevant features for similarity calculation
# features = ['danceability', 'energy', 'acousticness', 'tempo']
# X = df[features]
# X = df[['name', 'artist', 'tags', 'year', 'mode', 'acousticness', 'PCA_1']]
X = df

# Input field for song name
song_input = st.text_input("Enter a song name:")
# st.write(song_input)

# Recommendation function
def recommender(song_name, recommendation_set, model):
    # Use fuzzy matching to find the closest song name in the dataset
    idx=process.extractOne(song_name, recommendation_set['name'])[2]
    st.write(f"Song Selected: {df['name'][idx]} by {df['artist'][idx]}")

    # requiredSongs = recommendation_set.select_dtypes(np.number).copy()
    requiredSongs = recommendation_set.select_dtypes(np.number).drop(columns = ['cat','cluster','year']).copy()
    
    # Find 5 nearest neighbors using KNN
    distances, indices = model.kneighbors(requiredSongs.iloc[idx].values.reshape(1,-1))
    
    # Display the recommended songs
    st.write("Recommended Songs:")
    recommendations = []
    # for i in indices[0]:
    for i in indices:
        recommendations.append(f"{df['name'][i]} by {df['artist'][i]}")
    return recommendations

# If the user has entered a song name, perform the recommendation
if song_input:
    recommended_songs = recommender(song_input, X, knn10)
    st.write("\n".join(recommended_songs))
