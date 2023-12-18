
from flask import Flask, request, session, g, redirect,url_for, abort, render_template, flash
import pandas as pd
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

# Load and preprocess recipes data
def load_and_preprocess_data():
    recipes = pd.read_csv('recipes.csv', index_col=0)
    recipes = recipes[recipes['Ingredients'] != '[]']
    stop = set(stopwords.words('english'))
    recipes['Cleaned_Ingredients'] = recipes['Cleaned_Ingredients'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop]))
    recipes['Tokenized_Ingredients'] = recipes['Cleaned_Ingredients'].apply(word_tokenize)
    return recipes


recipes = load_and_preprocess_data()

# Initialize TF-IDF Vectorizer

vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, preprocessor=lambda x: x)
matrix = vectorizer.fit_transform(recipes['Tokenized_Ingredients'])

def compute_cosine_similarity(user_tokens):
    user_vector = vectorizer.transform([user_tokens])
    cosine_scores = cosine_similarity(user_vector, matrix)
    recipes['cosine_score'] = cosine_scores[0]
    return recipes

def recommend_recipes(user_ingredients):
    user_tokens = [ingredient.lower().strip() for ingredient in user_ingredients.split(',')]
    compute_cosine_similarity(user_tokens)
    recipes['match_count'] = recipes['Tokenized_Ingredients'].apply(lambda x: sum(1 for token in user_tokens if token in x))
    top_recipes = recipes.sort_values(by=['match_count', 'cosine_score'], ascending=[False, False]).head(5)
    # Convert DataFrame to a list of dictionaries for JSON serialization
    return top_recipes[['Title', 'Ingredients', 'Instructions']].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    ingredients = data.get('ingredients')
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    try:
        recommendations = recommend_recipes(ingredients)
        return jsonify(recommendations)
    except Exception as e:
        # Log the exception e
        return jsonify({"error": "An error occurred during the recommendation process"}), 500


@app.route('/')
def landing_page():
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
CORS(app)
