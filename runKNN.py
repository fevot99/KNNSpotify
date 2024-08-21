# python -m streamlit run runKNN.py

# Importing necessary libraries
import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name to get similar song recommendations based on music content similarity")

# Dropdown for selecting User ID with a blank initial state
user_id = st.selectbox(
    'Please log in with your User ID',
    options=[''] + [1001, 1002, 1003, 1004, 1005],  # The first option is a blank string
    index=0  # Set the default index to 0 to show the blank option initially
)

# Load your preprocessed dataset
df = pd.read_csv('preprocessed_data.csv')  # Preprocessed music data with numerical features

def recommender(song_name, recommendation_set):
    # Find the index of the song using fuzzy matching
    idx = process.extractOne(song_name, recommendation_set['name'])[2]
    st.write(f"Song Selected: {recommendation_set['name'][idx]} by {recommendation_set['artist'][idx]}")
    st.write(f"Music Genre Information: {recommendation_set['tags'][idx]}")
     
    # Determine the cluster of the selected song
    query_cluster = recommendation_set['cluster'][idx]

    # Filter the dataset to include only points from the same cluster
    filtered_data = recommendation_set[recommendation_set['cluster'] == query_cluster]

    # Reset the index of the filtered data for consistency
    filtered_data = filtered_data.reset_index(drop=True)
    
    # Attempt to find the index of the selected song within the filtered dataset
    try:
        new_idx = filtered_data[filtered_data['name'] == recommendation_set['name'][idx]].index[0]
    except IndexError:
        raise IndexError("The selected song is not found within the filtered cluster data.")

    # Find nearest 10 neighbors, let knn decide algo based on data
    knn10 = NearestNeighbors(metric='euclidean', algorithm='auto', n_neighbors=11) # Add 1 to account for the selected song itself

    # KNN Model
    model = knn10
    features = filtered_data.select_dtypes(np.number).drop(columns=['year', 'cluster'])
    model.fit(features)

    # Convert the query point to a DataFrame with the same column names as `features`
    query_point_filtered = pd.DataFrame([features.iloc[new_idx]], columns=features.columns)

    # Find the k nearest neighbors within the same cluster
    distances, indices = model.kneighbors(query_point_filtered)

    # Prepare recommendations
    recommendations = []
    for i in indices[0]:
        if i != new_idx:  # Exclude the selected song itself
            recommendations.append({
                'name': filtered_data.iloc[i]['name'],
                'artist': filtered_data.iloc[i]['artist'],
                'tags': filtered_data.iloc[i]['tags']
            })

    rec_df = pd.DataFrame(recommendations)
    return rec_df, idx

# Initialize session state for playlist
if 'playlist' not in st.session_state:
    st.session_state.playlist = pd.DataFrame(columns=['Song', 'Artist', 'Music Genre Tags', 'Original Song'])

# Input field for song name
song_name_input = st.text_input("Enter a song that you like:")

if song_name_input:
    table_df, original_song_idx = recommender(song_name_input, df)
    st.write("Here are some recommended songs that you may like:")
    
    # Reset index and drop the old index column
    table_df_reset = table_df.head(10).reset_index(drop=True)
    
    # Display the DataFrame without the index column
    st.dataframe(table_df_reset, use_container_width=True, hide_index=True)
    
    filtered_df = table_df_reset.iloc[1:11].reset_index(drop=True)
    
    # Display the filtered table with checkboxes for selection
    st.write("You may select any recommended songs below and click on the 'Add to Playlist' button to create your personal playlist")

    # Display the filtered DataFrame with checkboxes
    selected_indices = []
    for idx, row in filtered_df.iterrows():
        song_name = row.get('name', 'Unknown Song')
        artist_name = row.get('artist', 'Unknown Artist')
        if st.checkbox(f"{song_name} by {artist_name}", key=idx):
            selected_indices.append(idx)

    # Filter selected songs
    selected_songs = filtered_df.loc[selected_indices]

# If the user clicks the "Add to Playlist" button, show the selected songs
if st.button('Add to Playlist'):
    if not selected_songs.empty:
        # Add the original song to the playlist DataFrame
        original_song = pd.DataFrame([{
            'Song': df['name'][original_song_idx],
            'Artist': df['artist'][original_song_idx],
            'Music Genre Tags': df['tags'][original_song_idx],
            'Original Song': True
        }])
        
        # Add selected songs
        selected_songs = selected_songs.rename(columns={'name': 'Song', 'artist': 'Artist', 'tags': 'Music Genre Tags'})
        selected_songs['Original Song'] = False
        
        # Combine the original song and selected songs
        final_playlist = pd.concat([original_song, selected_songs], ignore_index=True)

        # Display the playlist
        st.write("Your Playlist:")
        st.dataframe(final_playlist, use_container_width=True, hide_index=True)
        
        # Save the updated playlist to a CSV file
        if user_id:
            filename = f'Playlist_{user_id}.csv'
            final_playlist.to_csv(filename, index=False)
            st.write(f"Playlist saved as {filename}")
        else:
            st.write("User ID is not set. Cannot save playlist.")
    else:
        st.write("No songs selected, please try again.")

