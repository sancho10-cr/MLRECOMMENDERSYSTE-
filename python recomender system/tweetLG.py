import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast
import re


def preprocess_tweet_data():
    df = pd.read_csv('sample.csv')  # Change the file name to your dataset file
    df['created_at'] = pd.to_datetime(df['created_at'])  # Convert 'created_at' column to datetime
    df['text'] = df['text'].astype(str)  # Ensure 'text' column is string type
    df['inbound'] = df['inbound'].astype(bool)  # Convert 'inbound' column to boolean
    return df


def preprocess_text_features(df):
    tfidf = TfidfVectorizer(stop_words='english')
    text_features = tfidf.fit_transform(df['text'])
    return text_features, tfidf


def build_lgbm_model(X_train, y_train):
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(X_train, y_train)
    return lgbm_model


def train_and_evaluate_models(df):
    text_features, tfidf = preprocess_text_features(df)

    label_encoder_lgbm = LabelEncoder()
    label_encoder_lgbm.fit(df['text'])

    X_train, _, y_train, _ = train_test_split(text_features, df['text'], test_size=0.2, random_state=42)

    lgbm_model = build_lgbm_model(X_train, y_train)

    return lgbm_model, tfidf, label_encoder_lgbm


def preprocess_tweet_text(tweet_text):
    processed_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', tweet_text).lower()
    return processed_text


def recommend_tweets(user_input, model, tweet_data, tfidf, label_encoder_lgbm):
    filtered_tweets = tweet_data[tweet_data['inbound']]

    recommendations = {}
    if not filtered_tweets.empty:
        input_text = filtered_tweets['text'].tolist()
        processed_input = [preprocess_tweet_text(text) for text in input_text]
        processed_user_input = preprocess_tweet_text(user_input)

        similarities = cosine_similarity(tfidf.transform([processed_user_input]), tfidf.transform(processed_input))

        similar_tweets_indices = np.argsort(similarities[0])[::-1][:5]

        model_recommendations = list(filtered_tweets.iloc[similar_tweets_indices]['text'])
        recommendations['lgbm'] = model_recommendations
    else:
        recommendations = {'lgbm': []}

    return recommendations


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tweet Recommendation System")

    tweet_data = preprocess_tweet_data()
    model, tfidf, label_encoder_lgbm = train_and_evaluate_models(tweet_data)

    label_tweet = tk.Label(root, text="Enter Tweet:")
    label_tweet.pack()

    entry_tweet = tk.Entry(root, width=50)
    entry_tweet.pack()

    def get_recommendation():
            user_input_tweet = entry_tweet.get()

            if user_input_tweet:
                recommendations = recommend_tweets(
                    user_input_tweet, model, tweet_data, tfidf, label_encoder_lgbm
                )

                result_text = "Top Recommended Tweets:\n"
                for model_name, model_recommendations in recommendations.items():
                    for i, tweet in enumerate(model_recommendations, start=1):
                        result_text += f"{i}. {tweet}\n"

                result_label.config(text=result_text)
            else:
                result_label.config(text="Please enter a tweet.")


    button = tk.Button(root, text="Recommend", command=get_recommendation)
    button.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()
