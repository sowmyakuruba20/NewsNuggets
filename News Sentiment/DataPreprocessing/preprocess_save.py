import logging
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure that all necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.wordnet = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)

    def preprocess_text(self, text):
        """Process text through various preprocessing steps."""
        text = self.remove_urls(text)
        text = self.remove_punctuations(text)
        text = self.to_lowercase(text)
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)

    def remove_urls(self, text):
        pattern = re.compile(r'http\S+|www\S+|https\S+')
        return re.sub(pattern, '', text)

    def remove_punctuations(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def to_lowercase(self, text):
        return text.lower()

    def tokenize_text(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens):
        return [self.wordnet.lemmatize(token) for token in tokens]

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.logger = logging.getLogger(__name__)

    def fit(self, texts):
        """Fit the TF-IDF vectorizer with the preprocessed texts."""
        self.vectorizer.fit(texts)

    def transform(self, texts):
        """Transform texts to TF-IDF vectors."""
        return self.vectorizer.transform(texts)

def save_objects():
    """Instantiate and save the preprocessor and vectorizer to pickle files."""
    preprocessor = TextPreprocessor()
    vectorizer = Vectorizer()

    # You would fit the vectorizer here with preprocessed texts if you have training data at this stage
    # For demonstration, we skip fitting here

    # Save preprocessor and vectorizer
    with open('text_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer.vectorizer, f)

    print("Preprocessor and vectorizer have been saved.")

if __name__ == "__main__":
    save_objects()
