import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocess_movie_data():
    # Load movie data
    df1 = pd.read_csv('movie recomender\\tmdb_5000_credits.csv')
    df2 = pd.read_csv('movie recomender\\tmdb_5000_movies.csv')
    df3 = df2.merge(df1, on='movie_id')
    df = df3.head(500)

    return df

def preprocess_text_features(df):
    # Preprocess text features
    tfidf = TfidfVectorizer(stop_words='english')
    text_features = tfidf.fit_transform(df['original_title'].fillna(''))
    return text_features

def build_svm_model(X_train, y_train):
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model

def build_logistic_regression_model(X_train, y_train):
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    return log_reg_model

def build_lightgbm_model(X_train, y_train):
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(X_train, y_train)
    return lgbm_model

def train_and_evaluate_models(df):
    # Preprocess the movie data
    text_features = preprocess_text_features(df)
    X_train, X_test, y_train, y_test = train_test_split(text_features, df['original_title'], test_size=0.2, random_state=42)

    models = {
        'SVM': build_svm_model(X_train, y_train),
        'Logistic Regression': build_logistic_regression_model(X_train, y_train),
        'LightGBM': build_lightgbm_model(X_train, y_train),
    }

    results = {'Test Accuracy': [], 'Train Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'Train Time': []}

    for model_name, model in models.items():
        # Train the model and record time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions and evaluations
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')

        # Append results to the dictionary
        results['Test Accuracy'].append(test_accuracy)
        results['Train Accuracy'].append(train_accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-Score'].append(f1)
        results['Train Time'].append(train_time)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix for {model_name}:\n{conf_matrix}")

    # Display results
    print("\nResults:")
    for metric, values in results.items():
        print(f"{metric}: {values}")

    # Plot barcharts
    plt.figure(figsize=(10, 6))
    plt.bar(models.keys(), results['Test Accuracy'], color='blue', alpha=0.7, label='Test Accuracy')
    plt.bar(models.keys(), results['Train Time'], color='orange', alpha=0.7, label='Train Time')
    plt.xlabel('Algorithms')
    plt.ylabel('Score/Time')
    plt.title('Test Accuracy and Train Time Comparison')
    plt.legend()
    plt.show()

    # Other necessary plots and comparisons can be added here
        # Plot barcharts
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Test Accuracy subplot
    axs[0].bar(models.keys(), results['Test Accuracy'], color='blue', alpha=0.7, label='Test Accuracy')
    axs[0].set_ylabel('Test Accuracy')
    axs[0].set_title('Test Accuracy Comparison')

    # Train Time subplot
    axs[1].bar(models.keys(), results['Train Time'], color='orange', alpha=0.7, label='Train Time')
    axs[1].set_ylabel('Train Time (seconds)')
    axs[1].set_title('Train Time Comparison')

    plt.xlabel('Algorithms')
    plt.show()


if __name__ == "__main__":
    movie_data = preprocess_movie_data()
    train_and_evaluate_models(movie_data)

