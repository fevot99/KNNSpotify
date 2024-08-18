# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name and get 5 similar song recommendations based on content similarity.")

# Load your preprocessed dataset (assuming you have a dataframe `df` with 'song', 'artist', and feature columns)
# Example DataFrame structure:
df = pd.read_csv('50000 Spotify Songs.csv') # Preprocessed music data with numerical features

# Load the trained KNN model from the pickle file
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# Use the relevant features for similarity calculation
features = ['danceability', 'energy', 'acousticness', 'tempo']
X = df[features]

# Input field for song name
song_name = st.text_input("Enter a song name:")

# Recommendation function
def recommender(song_name, df, model):
    # Use fuzzy matching to find the closest song name in the dataset
    idx = process.extractOne(song_name, df['song'])[2]
    st.write(f"Song Selected: {df['song'][idx]} by {df['artist'][idx]}")
    
    # Find 5 nearest neighbors using cosine similarity
    distances, indices = model.kneighbors(X.iloc[idx].values.reshape(1, -1))
    
    # Display the recommended songs
    st.write("Recommended Songs:")
    recommendations = []
    for i in indices[0]:
        recommendations.append(f"{df['song'][i]} by {df['artist'][i]}")
    return recommendations

# If the user has entered a song name, perform the recommendation
if song_name:
    recommended_songs = recommender(song_name, df, knn10)
    st.write("\n".join(recommended_songs))
