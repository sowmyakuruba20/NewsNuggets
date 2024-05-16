import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

class Vectorizer:
    def __init__(self):
        """
        Initialize the Vectorizer instance.
        """
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bow_vectorizer = CountVectorizer()
        self.logger = logging.getLogger(__name__)  # Use the global logger

    def fit_transform_tfidf(self, X_train, X_test):
        """
        Fit and transform text data to TF-IDF vectors.

        Args:
        - X_train (list): List of preprocessed training text data.
        - X_val (list): List of preprocessed validation text data.
        - X_test (list): List of preprocessed test text data.

        Returns:
        - tuple: A tuple containing TF-IDF vectors for training, validation, and test data.
        """
        self.logger.info("Fitting and transforming text data to TF-IDF vectors.")
        # Convert list of tokens to string
        X_train_text = [' '.join(tokens) for tokens in X_train]
        X_test_text = [' '.join(tokens) for tokens in X_test]

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test_text)

        # Save TF-IDF vectorizer model
        self.save_vectorizer_model(self.tfidf_vectorizer, 'tfidf_vectorizer.pkl')

        self.logger.info("TF-IDF vectorization completed.")
        return X_train_tfidf, X_test_tfidf

    def fit_transform_bow(self, X_train, X_test):
        """
        Fit and transform text data to Bag of Words (BoW) vectors.

        Args:
        - X_train (list): List of preprocessed training text data.
        - X_val (list): List of preprocessed validation text data.
        - X_test (list): List of preprocessed test text data.

        Returns:
        - tuple: A tuple containing Bag of Words (BoW) vectors for training, validation, and test data.
        """
        self.logger.info("Fitting and transforming text data to Bag of Words (BoW) vectors.")
        # Convert list of tokens to string
        X_train_text = [' '.join(tokens) for tokens in X_train]
        X_test_text = [' '.join(tokens) for tokens in X_test]

        X_train_bow = self.bow_vectorizer.fit_transform(X_train_text)
        X_test_bow = self.bow_vectorizer.transform(X_test_text)

        # Save Bag of Words (BoW) vectorizer model
        self.save_vectorizer_model(self.bow_vectorizer, 'bow_vectorizer.pkl')

        self.logger.info("Bag of Words (BoW) vectorization completed.")
        return X_train_bow, X_test_bow

    def save_vectorizer_model(self, vectorizer, filename):
        """
        Save the vectorizer model to a file.

        Args:
        - vectorizer: The vectorizer model to be saved.
        - filename (str): Name of the file to save the model.
        """
        model_filename = filename
        with open(model_filename, 'wb') as model_file:
            pickle.dump(vectorizer, model_file)
        self.logger.info(f"Vectorizer model saved as {model_filename}.")

    def load_vectorizer_model(self, filename):
        """
        Load a vectorizer model from a file.

        Args:
        - filename (str): Name of the file containing the saved model.

        Returns:
        - vectorizer: The loaded vectorizer model.
        """
        model_filename = filename
        with open(model_filename, 'rb') as model_file:
            vectorizer = pickle.load(model_file)
        self.logger.info(f"Vectorizer model loaded from {model_filename}.")
        return vectorizer
