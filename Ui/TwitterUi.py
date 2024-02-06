import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load your dataset (replace with the actual dataset file path)
df = pd.read_csv('movie recomender\\Tweets.csv', encoding='latin-1', low_memory=False)

# Create TF-IDF vectors for tweet texts
df['text'] = df['text'].fillna('')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Function to remove URLs from text
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Function to recommend tweets and UserIDs
def recommend_tweets_and_userids(input_text, num_recommendations=5):
    # Remove URLs from the input text
    input_text = remove_urls(input_text)

    # Transform the input text into a TF-IDF vector
    input_vector = tfidf_vectorizer.transform([input_text])

    # Calculate cosine similarities between the input text and all tweets
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)

    # Get indices of tweets with the highest cosine similarities
    tweet_indices = cosine_similarities.argsort()[0][::-1][:num_recommendations]

    # Retrieve the recommended tweets and UserIDs
    recommended_tweets = df.loc[tweet_indices, 'text'].tolist()
    recommended_userids = df.loc[tweet_indices, 'UserID'].tolist()

    return recommended_tweets, recommended_userids

# Create a Tkinter GUI
def get_recommendations():
    input_text = tweet_entry.get("1.0", tk.END).strip()
    num_recommendations = int(num_recommended.get())

    if input_text:
        recommended_tweets, recommended_userids = recommend_tweets_and_userids(input_text, num_recommendations)
        num_tweets_label.config(text=f"Number of Tweets: {len(df)}")
        result_label.config(text=f"Top {num_recommendations} Recommended Tweets:")
        recommended_tweets_text.config(state=tk.NORMAL)
        recommended_tweets_text.delete(1.0, tk.END)
        for i, (tweet, userid) in enumerate(zip(recommended_tweets, recommended_userids), start=1):
            recommended_tweets_text.insert(tk.END, f"{i}: UserID: {userid}, {tweet}\n")
        recommended_tweets_text.config(state=tk.DISABLED)
    else:
        result_label.config(text="Please enter a tweet text.")

# Create the main window
root = tk.Tk()
root.title("Tweet Recommender")

# Label to display the number of tweets in the dataset
num_tweets_label = tk.Label(root, text=f"Number of Tweets: {len(df)}")
num_tweets_label.pack()

# Text widget for entering tweet text
tweet_label = tk.Label(root, text="Enter Tweet Text:")
tweet_label.pack()
tweet_entry = tk.Text(root, height=5, width=40)
tweet_entry.pack()

# Entry for specifying the number of recommendations
num_recommendations_label = tk.Label(root, text="Number of Recommendations:")
num_recommendations_label.pack()
num_recommended = tk.StringVar(value="5")  # Default value
num_recommendations_entry = ttk.Combobox(root, textvariable=num_recommended, values=["1", "2", "3", "4", "5"])
num_recommendations_entry.pack()

# Button to get tweet recommendations
recommend_button = tk.Button(root, text="Get Recommendations", command=get_recommendations)
recommend_button.pack()

# Label to display the recommendations
result_label = tk.Label(root, text="")
result_label.pack()

# Text widget to display recommended tweets (read-only)
recommended_tweets_text = tk.Text(root, height=40, width=60, state=tk.DISABLED)
recommended_tweets_text.pack()

# Start the main loop
root.mainloop()
