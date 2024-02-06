# Importing necessary modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load and preprocess the dataset
dataset_path = "D:\\codes\\python\\Recomendation\\tweet_Recondation_system\\sample.csv"
df = pd.read_csv(dataset_path)
tweets = df["text"].tolist()
user_ids = df["author_id"].tolist()
labels = df["inbound"].tolist()

user_encoder = LabelEncoder()
label_encoder = LabelEncoder()
user_ids_encoded = user_encoder.fit_transform(user_ids)
labels_encoded = label_encoder.fit_transform(labels)

scaler = MinMaxScaler()
user_ids_scaled = scaler.fit_transform(user_ids_encoded.reshape(-1, 1))

tweets = tweets[:500]
user_ids_scaled = user_ids_scaled[:500]
labels_encoded = labels_encoded[:500]

# Combine user IDs and text data into a feature matrix
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(tweets)
feature_matrix = np.hstack([user_ids_scaled, text_features.toarray()])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels_encoded, test_size=0.3, random_state=42)

# SARSA agent
class SARSAgent:
    def __init__(self, num_actions, num_features):
        self.num_actions = num_actions
        self.num_features = num_features
        self.q_table = np.zeros((num_features, num_actions))
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state, next_action, alpha=0.1, gamma=0.99):
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * next_q)
        self.q_table[state, action] = new_q

# Using the SARSA agent
num_actions = len(np.unique(y_train))
num_features = feature_matrix.shape[1]

# sarsa_agent = SARSAgent(num_actions, num_features)

# num_episodes = 100
# for episode in range(num_episodes):
#     state = int(X_train[0, 0])
#     action = sarsa_agent.choose_action(state)
#     for i in range(1, X_train.shape[0]):
#         next_state = int(X_train[i, 0])
#         next_action = sarsa_agent.choose_action(next_state)
#         reward = int(y_train[i] == next_action)
#         sarsa_agent.update_q_table(state, action, reward, next_state, next_action)
#         state = next_state
#         action = next_action

# # Results and Evaluations
# predictions = [sarsa_agent.choose_action(int(state)) for state in X_test[:, 0]]
# accuracy = accuracy_score(y_test, predictions)

# print(f"Accuracy {accuracy}:")

# Define the hyperparameters to test
alphas = [0.1, 0.01, 0.001,0.03,0.9,0.78,0.9]
gammas = [0.9, 0.95, 0.99,0.1,0.3]
num_episodes_list = [50, 100, 150,200,500,700,1000,]
epsilon =[ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]
# Initialize best parameters and accuracy
best_alpha = None
best_gamma = None
best_num_episodes = None
best_accuracy = 0.00
epsilon=None
# Loop over the hyperparameters
for alpha in alphas:
    for gamma in gammas:
        for num_episodes in num_episodes_list:
            # Train the SARSA agent
            sarsa_agent = SARSAgent(num_actions, num_features)
            sarsa_agent.epsilon = 0.1 # Exploration rate 
            for episode in range(num_episodes):
                state = int(X_train[0, 0])
                action = sarsa_agent.choose_action(state)
                for i in range(1, X_train.shape[0]):
                    next_state = int(X_train[i, 0])
                    next_action = sarsa_agent.choose_action(next_state)
                    reward = int(y_train[i] == next_action)
                    sarsa_agent.update_q_table(state, action, reward, next_state, next_action, alpha, gamma)
                    state = next_state
                    action = next_action
            
            # Evaluate the SARSA agent
            predictions = [sarsa_agent.choose_action(int(state)) for state in X_test[:, 0]]
            accuracy = accuracy_score(y_test, predictions)
            
            # Update best parameters if current model is better
            if accuracy > best_accuracy:
                best_alpha = alpha
                best_gamma = gamma
                best_num_episodes = num_episodes
                best_accuracy = accuracy

print(f"Best Alpha: {best_alpha}, Best Gamma: {best_gamma}, Best num_episodes: {best_num_episodes}")
print(f"Best Accuracy {best_accuracy}:")
