import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# Load dataset
df = pd.read_csv('/Users/varunreddy/Desktop/My project/main.py/data/products.csv')

# Drop non-feature columns
X = df.drop(['product_id', 'product_name'], axis=1)

# Train NearestNeighbors model
model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
model.fit(X)

# Save model as pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to app/model.pkl")
