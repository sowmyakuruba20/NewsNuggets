# NEWSNUGGETS Application

NEWSNUGGETS is a cohesive web-based application designed to analyze news articles using machine learning models. It integrates three key functionalities: News Article Sentiment Classification, News Article Category Classification, and News Article Summarization. Given a user's text input (a news article), NEWSNUGGETS provides sentiment analysis, categorization, and a concise summary of the content.

## Project Structure

- `Application/`: Main folder.
  - `app.py`: Flask application that loads models and vectorizers, handles web requests and renders results.
  - `models.py`: Contains classes for text preprocessing, vectorization, and summarization.
  - `templates/`: Folder containing HTML templates for the user interface.
    - `index.html`: The main page where users input the text of a news article.
    - `result.html`: Displays the analysis results, including category, sentiment, and summary.
  - `Pickle Files`: Stores pre-trained models and vectorizers.
    - `tfidf_vectorizer.pkl1`: TF-IDF vectorizer for category classification.
    - `logistic_regression_model.pkl`: Model for category classification.
    - `tfidf_vectorizer_sent.pkl1`: TF-IDF vectorizer for sentiment analysis.
    - `logistic_regression_model_sent.pkl`: Model for sentiment analysis.
    - `news_summarizer.pkl`: Summarization model.

## Features

- **Sentiment Classification**: Determines if the sentiment of the news article is positive or negative using logistic regression.
- **Category Classification**: Classifies the news article into predefined categories.
- **Summarization**: Generates a concise summary of the news article.

## Prerequisites

- Python 3.x
- Flask
- NLTK
- scikit-learn
- Pickle

## Setup Instructions

1. Ensure you have Python installed on your system.
2. Clone or download the NEWSNUGGETS application folder.

3. Navigate to the application directory:
cd Application

4. Install the required Python libraries:
pip install Flask nltk scikit-learn pickle-mixin

5. Make sure NLTK resources are downloaded:
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

5. Run the application by executing the following command in your terminal:
python app.py


This will start a local web server. By default, the Flask app will be available at http://127.0.0.1:5000 in your web browser.

## Using the Application

- Open your web browser and navigate to http://127.0.0.1:5000.
- Enter the text of a news article into the provided text area.
- Submit the text to receive a sentiment analysis, category classification, and a summarized version of the article.

## Technologies Used

- **Flask**: Serves the web application and handles backend logic.
- **NLTK**: Used for text preprocessing and summarization techniques.
- **scikit-learn**: Implements machine learning models for text classification and vectorization.

## Authors:

- [ANITHA BALACHANDRAN, MADHURA BHATSOORI, SOWJANYA PAMULAPATI,
SOWMYA KURUBA, SWATHI RAMESH ]
