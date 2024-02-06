import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast
import re


def preprocess_movie_data():
    df1 = pd.read_csv('movie recomender\\tmdb_5000_credits.csv')
    df2 = pd.read_csv('movie recomender\\tmdb_5000_movies.csv')
    df1.columns = ['movie_id', 'title', 'cast', 'crew']
    df = df2.merge(df1, on='movie_id')
    df = df[['genres', 'original_title', 'vote_average']]
    df['vote_average'] = df['vote_average'].fillna(0)
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x))  
    return df


def preprocess_text_features(df):
    tfidf = TfidfVectorizer(stop_words='english')
    text_features = tfidf.fit_transform(df['original_title'].fillna(''))
    return text_features, tfidf


def build_pipeline_svm(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classify', SVC(kernel='linear', probability=True))
    ]), label_encoder


def train_and_evaluate_models(df):
    text_features, tfidf = preprocess_text_features(df)

    label_encoder_svm = LabelEncoder()
    label_encoder_svm.fit(df['original_title'])

    X_train, _, y_train, _ = train_test_split(text_features, df['original_title'], test_size=0.2, random_state=42)

    svm_model, label_mapping_svm = build_pipeline_svm(X_train, y_train)

    return {'svm': svm_model}, tfidf, label_mapping_svm



def preprocess_movie_titles(movie_titles):
    processed_titles = []
    for title in movie_titles:
        processed_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title).lower()
        processed_titles.append(processed_title)
    return processed_titles
# Existing code...

def recommend_movies(user_input, genres, models, movie_data, tfidf, label_mapping_svm):
    filtered_movies = movie_data[movie_data['genres'].apply(lambda x: any(item['name'] in genres for item in x))]

    recommendations = {}
    if not filtered_movies.empty:
        for model_name, model in models.items():
            if model_name == 'svm':
                input_text = filtered_movies['original_title'].fillna('').tolist()
                processed_input = preprocess_movie_titles(input_text)
                model.fit(processed_input, filtered_movies['original_title'])

                processed_user_input = preprocess_movie_titles([user_input])[0]
                predictions = model.predict([processed_user_input])  # Get the prediction output

                try:
                    predicted_movie_label = predictions
                except ValueError as e:
                    # Handle unseen label case
                    unseen_label = processed_user_input

                    if unseen_label not in label_mapping_svm.classes_:
                        # Add the unseen label to the label encoder
                        label_mapping_svm.classes_ = np.append(label_mapping_svm.classes_, unseen_label)

                    # Try inverse transform again
                    predicted_movie_label = label_mapping_svm.inverse_transform(predictions)

                # Convert predicted label to string for direct use in cosine similarity
                predicted_movie_str = str(predicted_movie_label[0])

                similarities = cosine_similarity(tfidf.transform([predicted_movie_str]), tfidf.transform(filtered_movies['original_title']))

                # Get the indices of movies similar to the predicted movie based on cosine similarity
                similar_movies_indices = np.argsort(similarities[0])[::-1][:5]  # Consider top 5 similar movies

                model_recommendations = list(filtered_movies.iloc[similar_movies_indices]['original_title'])
                recommendations[model_name] = model_recommendations
    else:
        recommendations = {model_name: [] for model_name in models}

    return recommendations

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Movie Recommendation System")

    movie_data = preprocess_movie_data()
    models, tfidf, label_mapping_svm = train_and_evaluate_models(movie_data)

    label_title = tk.Label(root, text="Enter Movie Title:")
    label_title.pack()

    entry_title = tk.Entry(root, width=50)
    entry_title.pack()

    label_genres = tk.Label(root, text="Enter Genres:")
    label_genres.pack()

    entry_genres = tk.Entry(root, width=50)
    entry_genres.pack()

    def get_recommendation():
            user_input_title = entry_title.get()
            user_input_genres = entry_genres.get().split(",") if entry_genres.get() else []

            if user_input_title and user_input_genres:
                recommendations = recommend_movies(
                    user_input_title, user_input_genres, models, movie_data, tfidf, label_mapping_svm
                )

                result_text = f"Top Recommended 5 {str(user_input_genres)} Movies:\n"
                for model_name, model_recommendations in recommendations.items():
                    # result_text += f"{model_name}:\n"

                    # Display predicted label and its corresponding rating count
                    predicted_movie_str = ""  # Initialize variable

                    if model_recommendations:  # Check if recommendations are available
                        predicted_movie_str = model_recommendations[0]  # Take the first recommendation

                    #result_text += f"Predicted{user_input_genres} Movies: {predicted_movie_str}\n"

                    for i, movie in enumerate(model_recommendations, start=1):
                        result_text += f"{i}. {movie}\n"

                result_label.config(text=result_text)
            else:
                result_label.config(text="Please enter a movie title and genres.")


    button = tk.Button(root, text="Recommend", command=get_recommendation)
    button.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()
