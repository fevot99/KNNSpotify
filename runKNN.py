# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name and get 5 similar song recommendations based on content similarity.")

# Adding a sidebar
st.sidebar.title("Sidebar")
option = st.sidebar.selectbox(
    'Please enter your User ID',
    list(range(1001, 1006)))
   
# st.image('pic.jpg')
Age=st.sidebar.radio('Please enter your Age Group',options=['Under 20','20+','30+','40+','Over 50'])

# Load your preprocessed dataset
df = pd.read_csv('testpca.csv') # Preprocessed music data with numerical features

# Load the trained KNN model from the pickle file
with open('knn_model.pkl', 'rb') as f:
    knn10 = pickle.load(f)

# Use the relevant features for similarity calculation
# features = ['danceability', 'energy', 'acousticness', 'tempo']
# X = df[features]
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
    st.write("Song Name \ Artist \ Music Genre Tags\n")
    recommendations = []
    # for i in indices[0]:
    for i in indices:
        recommendations.append(f"{df['name'][i]} by {df['artist'][i]}")
        st.write(df['name'][i] + "   \   " + df['artist'][i] + "   \   " + df['tags'][i])

# If the user has entered a song name, perform the recommendation
if song_input:
    recommended_songs = recommender(song_input, X, knn10)
    # st.write("\n".join(recommended_songs), "\n")

# Adding a button
if st.button('Add to Playlist'):
    st.write('Song added to Playlist')
st.write('Tick the songs you like to add to Playlist')

# Adding a button
if st.button('Click to refresh playlist'):
    st.write('Updated Playlist')

st.write("Here is your current playlist:")
# Dataframe display
playlist_df = pd.DataFrame({
    'Songs': ["Moonlight Sonata", "Viva la Vida", "Toccata and Fugue in D Minor"],
    'Artist': ["Ludwig van Beethoven", "Coldplay", "Johann Sebastian Bach"]
    })
st.write(playlist_df)

# Create a slider
# rating = st.slider("Please rate the recommended song (5 being Highest", min_value=1, max_value=5, value=1)

# Display the selected value
# st.write("You have given a rating of ", rating)



