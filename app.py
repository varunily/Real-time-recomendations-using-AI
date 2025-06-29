from flask import Flask, request, jsonify
from Recomendations import Recommender

app = Flask(__name__)
recommender = Recommender()

@app.route('/')
def home():
    return "API is running!"

@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id', type=int)
    n = request.args.get('n', default=5, type=int)
    if product_id is None:
        return jsonify({'error': 'Missing product_id'}), 400
    try:
        recommendations = recommender.recommend(product_id, n)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)