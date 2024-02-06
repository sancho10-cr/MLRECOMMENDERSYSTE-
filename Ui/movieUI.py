import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk

# Load movie data
df1 = pd.read_csv('movie recomender\\tmdb_5000_credits.csv')
df2 = pd.read_csv('movie recomender\\tmdb_5000_movies.csv')
df1.columns = ['movie_id','tittle','cast','crew']
df2= df2.merge(df1,on='movie_id')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


# Function to get movie recommendations
def get_recommendations(title, num_recommendations=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(num_recommendations + 1)] if len(sim_scores) > 1 else []
    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = df2['title'].iloc[movie_indices].reset_index(drop=True)
    return recommended_titles


# Create a Tkinter GUI
def recommend_movies():
    input_movie = movie_entry.get()
    num_recommendations = int(num_recommended.get())

    if input_movie:
        recommendations = get_recommendations(input_movie, num_recommendations)
        result_label.config(text=f"Top {num_recommendations} Recommended Movies for '{input_movie}':")
        recommendation_text.config(text='\n'.join(recommendations))
    else:
        result_label.config(text="Please enter a movie title.")

# Create the main window
root = tk.Tk()
root.title("Movie Recommender")

# Label to display the number of movies available in the dataset
num_movies_label = tk.Label(root, text=f"Number of Movies in Dataset: {len(df2)}")
num_movies_label.pack()

# Entry for entering a movie title
movie_label = tk.Label(root, text="Enter a Movie Title:")
movie_label.pack()
movie_entry = tk.Entry(root)
movie_entry.pack()

# Entry for specifying the number of recommendations
num_recommendations_label = tk.Label(root, text="Number of Recommendations:")
num_recommendations_label.pack()
num_recommended = tk.StringVar(value="5")  # Default value
num_recommendations_entry = ttk.Combobox(root, textvariable=num_recommended, values=["1", "2", "3", "4", "5"])
num_recommendations_entry.pack()

# Button to get movie recommendations
recommend_button = tk.Button(root, text="Get Recommendations", command=recommend_movies)
recommend_button.pack()

# Label to display the recommendations
result_label = tk.Label(root, text="")
result_label.pack()

recommendation_text = tk.Label(root, text="", wraplength=400, justify="left")
recommendation_text.pack()

# Start the main loop
root.mainloop()
