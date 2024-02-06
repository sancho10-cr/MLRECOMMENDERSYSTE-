import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load and preprocess the dataset
# ... (same as in your original code)
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

tweets = tweets[:500000]
user_ids_scaled = user_ids_scaled[:500000]
labels_encoded = labels_encoded[:500000]
# Combine user IDs and text data into a feature matrix
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(tweets)
feature_matrix = np.hstack([user_ids_scaled.reshape(-1, 1), text_features.toarray()])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels_encoded, test_size=0.3, random_state=42)

# Build a Q-learning model
class QLearningAgent:
    def __init__(self, num_actions, num_features):
        self.num_actions = num_actions
        self.num_features = num_features
        self.q_table = np.zeros((num_features, num_actions))

    def choose_action(self, state):
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
        self.q_table[state, action] = new_q

# Train the Q-learning agent
num_users = len(user_encoder.classes_)
num_actions = 1000
num_features = feature_matrix.shape[1]

q_agent = QLearningAgent(num_actions, num_features)

num_episodes = 10000
for episode in range(num_episodes):
    for i in range(X_train.shape[0]):
        state = int(X_train[i, 0])  
        action = q_agent.choose_action(state)
        reward = int(y_train[i] == action)
        next_state = int(X_train[i, 0]) 
        q_agent.update_q_table(state, action, reward, next_state)

# Evaluate the Q-learning agent on the test set
predictions = [q_agent.choose_action(int(state)) for state in X_test[:, 0]]
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# ... (Previous code)

# Evaluate the Q-learning agent on the test set
predictions = [q_agent.choose_action(int(state)) for state in X_test[:, 0]]
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Plot ROC curve
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), predictions, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()








