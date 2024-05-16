# Import necessary packages
from flask import Flask, request, render_template
import pickle

# Import local classes for processing
from models import TextPreprocessor, Vectorizer, TextSummarizer

app = Flask(__name__)

# Load the vectorizers and models directly
with open('tfidf_vectorizer.pkl1', 'rb') as f:
    category_vectorizer = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    category_model = pickle.load(f)

with open('news_summarizer.pkl', 'rb') as f:
    summarizer = pickle.load(f)

with open('logistic_regression_model_sent.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

with open('tfidf_vectorizer_sent.pkl1', 'rb') as f:
    sentiment_vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        original_text = request.form['text']
        
        # Instantiate TextPreprocessor directly in the route to ensure it's correctly set up
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess_text(original_text)
        
        # Vectorize text for category prediction
        category_vectorized_text = category_vectorizer.transform([processed_text])
        
        # Predict category
        category_prediction = category_model.predict(category_vectorized_text)
        category = category_prediction[0]  # Adjust based on model's output
        
        # Vectorize text for sentiment analysis
        sentiment_vectorized_text = sentiment_vectorizer.transform([processed_text])
        
        # Predict sentiment
        sentiment_prediction = sentiment_model.predict(sentiment_vectorized_text)
        sentiment = sentiment_prediction[0]  # Adjust based on model's output

        # Generate summary
        summary = summarizer.run_summarization(original_text)

        # Redirect to the result page with the results
        return render_template('result.html', category=category, summary=summary, sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
