import pandas as pd
import pickle

class Recommender:
    def __init__(self, model_path='model.pkl', product_data='main.py/data/products.csv'):
        self.model = pickle.load(open(model_path, 'rb'))
        self.products = pd.read_csv(product_data)
        self.feature_columns = self.products.columns.difference(['product_id', 'product_name'])

    def recommend(self, product_id, n=5):
        if product_id not in self.products['product_id'].values:
            raise ValueError(f"Product ID {product_id} not found in dataset.")

        product_row = self.products[self.products['product_id'] == product_id]
        feature_vector = product_row[self.feature_columns]

        total_items = len(self.products)
        n_neighbors = min(n + 1, total_items)

        distances, indices = self.model.kneighbors(feature_vector, n_neighbors=n_neighbors)
        
        # Remove the product itself from the results
        recommended_indices = [i for i in indices[0] if self.products.iloc[i]['product_id'] != product_id][:n]

        recommendations = self.products.iloc[recommended_indices][['product_id', 'product_name']]
        return recommendations.to_dict(orient='records')
