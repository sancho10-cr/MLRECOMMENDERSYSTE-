# Import necessary libraries
import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import numpy as np
import ast
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Function for loading and preprocessing tweet data
def preprocess_tweet_data():
    # Load the tweet data from CSV file (replace 'sample.csv' with your actual file)
    df = pd.read_csv('sample.csv')

    # Basic preprocessing
    df['created_at'] = pd.to_datetime(df['created_at'])  # Convert 'created_at' column to datetime
    df['text'] = df['text'].astype(str)  # Ensure 'text' column is string type
    df['inbound'] = df['inbound'].astype(bool)  # Convert 'inbound' column to boolean

    return df


# Your simplified preprocessing function
def preprocess_text(tweet_text):
    # Lowercase the text
    processed_text = tweet_text.lower()
    return processed_text


# Function for training and evaluating models without advanced preprocessing
def train_and_evaluate_models(df):
    # Perform text preprocessing
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['inbound'], test_size=0.2,
                                                        random_state=42)

    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Transform text data into TF-IDF features
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Initialize and train LightGBM classifier with parameter tuning
    lgbm_model = LGBMClassifier()
    param_grid = {'n_estimators': [100, 200, 300],
                  'learning_rate': [0.05, 0.1, 0.2]}
    grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train)
    lgbm_model = grid_search.best_estimator_

    # Initialize and train SVM classifier with parameter tuning
    svm_model = SVC(kernel='linear', probability=True)
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train)
    svm_model = grid_search.best_estimator_

    # Evaluate models on test set
    lgbm_predictions = lgbm_model.predict(X_test_tfidf)
    svm_predictions = svm_model.predict(X_test_tfidf)

    # Print classification reports for both models
    print("LightGBM Model Evaluation:")
    print(classification_report(y_test, lgbm_predictions))
    print("SVM Model Evaluation:")
    print(classification_report(y_test, svm_predictions))

    return {'lgbm': lgbm_model, 'svm': svm_model}, tfidf


# Function for generating word cloud visualization
# Function for generating word cloud visualization with custom parameters
def generate_wordcloud(text):
    # Define parameters for WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, contour_color='steelblue',
                          colormap='viridis', contour_width=3).generate(text)
    
    # Set the size of the graph
    plt.figure(figsize=(12, 6))
    
    # Display the word cloud graph
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()



# Your enhanced UI code with simplified preprocessing

# Main function
if __name__ == "__main__":
    # Create the Tkinter GUI
    root = tk.Tk()
    root.title("Tweet Recommendation System")

    # Preprocess tweet data and train models
    tweet_data = preprocess_tweet_data()
    models, tfidf = train_and_evaluate_models(tweet_data)

    # Function for generating recommendations and visualizations
    def get_recommendation():
        # Get user input
        user_input_tweet = entry_tweet.get()

        if user_input_tweet:
            # Preprocess user input
            processed_user_input = preprocess_text(user_input_tweet)

            # Get recommendations for each model
            recommendations = {}
            for model_name, model in models.items():
                similarities = cosine_similarity(tfidf.transform([processed_user_input]),
                                                 tfidf.transform(tweet_data['processed_text']))
                similar_tweets_indices = np.argsort(similarities[0])[::-1][:5]
                model_recommendations = list(tweet_data.iloc[similar_tweets_indices]['text'])
                recommendations[model_name] = model_recommendations

                # Generate word cloud visualization for recommended tweets
                generate_wordcloud(" ".join(model_recommendations))

            # Display recommendations in the UI
            result_text = "Top Recommended Tweets:\n"
            for model_name, model_recommendations in recommendations.items():
                for i, tweet in enumerate(model_recommendations, start=1):
                    result_text += f"{model_name} Recommendation {i}: {tweet}\n"

            result_label.config(text=result_text)
        else:
            result_label.config(text="Please enter a tweet.")

    # UI layout
    label_tweet = tk.Label(root, text="Enter Tweet:")
    label_tweet.pack()

    entry_tweet = tk.Entry(root, width=50)
    entry_tweet.pack()

    button = tk.Button(root, text="Recommend", command=get_recommendation)
    button.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    # Run the Tkinter event loop
    root.mainloop()
