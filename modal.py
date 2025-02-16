from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd

# Load dataset (replace with actual dataset)
data = pd.read_csv("spotify_data.csv")

# Select features and labels
X = data[['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
y = data['genre']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
pickle.dump(model, open("model.pkl", "wb"))
