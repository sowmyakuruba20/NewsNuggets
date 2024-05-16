import logging
from DataLoader.data_loader import DataLoader
from DataPreprocessing.data_preprocessing import TextPreprocessor
from TextVectorizer.vectorizer import Vectorizer
from Models.train_val_test import ModelTrain
from skopt.space import Real, Categorical, Integer
import os

# Set up logging configuration to capture all events
def setup_logging(log_file_path):
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main():
    try:
        # Setting up logging
        # Path needs adjustment based on your local environment setup
        log_file_path = '/Users/anithabalachandran/Data Analytics/Third sem/Data 240/Project_NewsNuggets/Category_Classification/logs/application.log'
        setup_logging(log_file_path)
        logging.info("Main process started")

        # Path to the dataset
        # Path needs adjustment based on your local environment setup
        data_path = "/Users/anithabalachandran/Data Analytics/Third sem/Data 240/Project_NewsNuggets/Category_Classification/Dataset/compiled_news_data.csv"
        data_loader = DataLoader(data_path)
        dataset = data_loader.load_data()

        if dataset is None or dataset.empty:
            raise ValueError("Dataset is empty or None. Check the data loading process.")

        logging.info("Data loaded successfully")

        # Initializing and using text preprocessor
        # Path needs adjustment based on your local environment setup
        output_dir = "/Users/anithabalachandran/Data Analytics/Third sem/Data 240/Project_NewsNuggets/Category_Classification/Dataset/"
        preprocessor = TextPreprocessor(output_dir=output_dir)
        X_train, X_test, y_train, y_test = preprocessor.preprocess_text(dataset, 'Content', 'Category', save_to_csv=True)
        logging.info("Data preprocessing completed and saved to CSV files")

        # Define Bayesian optimization search spaces for model tuning
        search_spaces = {
            'lr': {'penalty': Categorical(['l2']), 'C': Real(0.1, 10, prior='log-uniform'), 'solver': Categorical(['liblinear', 'lbfgs']), 'max_iter': Integer(100, 200)},
            'dt': {'max_depth': Integer(10, 50), 'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(1, 4)},
            'rf': {'n_estimators': Integer(50, 200), 'max_depth': Integer(10, 50), 'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(1, 4)},
            'ab': {'n_estimators': Integer(50, 200), 'learning_rate': Real(0.1, 1, prior='log-uniform')},
            'nb': {'alpha': Real(1e-2, 1e+0, prior='log-uniform')}
        }

        # Initialize vectorizer
        vectorizer = Vectorizer()
        X_train_tfidf, X_test_tfidf = vectorizer.fit_transform_tfidf(X_train, X_test)
        vectorizer.save_vectorizer_model(vectorizer.tfidf_vectorizer, 'tfidf_vectorizer.pkl')  # Save TF-IDF vectorizer model
        logging.info("TF-IDF models trained and evaluated successfully")

        X_train_bow, X_test_bow = vectorizer.fit_transform_bow(X_train, X_test)
        vectorizer.save_vectorizer_model(vectorizer.bow_vectorizer, 'bow_vectorizer.pkl')  # Save BoW vectorizer model
        logging.info("Bag of Words (BoW) vectorization completed")
        logging.info("Vectorization completed")

        # Initialize and train models using TF-IDF vectorized data
        # Path needs adjustment based on your local environment setup
        output_dir_tfidf = '/Users/anithabalachandran/Data Analytics/Third sem/Data 240/Project_NewsNuggets/Category_Classification/Results/TFIDF/'
        model_trainer_tfidf = ModelTrain(output_dir_tfidf, 'TFIDF')
        for model_name, search_space in search_spaces.items():
            model_trainer_tfidf.train_model(model_name, X_train_tfidf, y_train, X_test_tfidf, y_test, search_space)
        model_trainer_tfidf.finalize_results()

        # Initialize and train models using BoW vectorized data
        # Path needs adjustment based on your local environment setup
        output_dir_bow = '/Users/anithabalachandran/Data Analytics/Third sem/Data 240/Project_NewsNuggets/Category_Classification/Results/BOW/'
        model_trainer_bow = ModelTrain(output_dir_bow, 'BoW')
        for model_name, search_space in search_spaces.items():
            model_trainer_bow.train_model(model_name, X_train_bow, y_train, X_test_bow, y_test, search_space)
        model_trainer_bow.finalize_results()
        logging.info("BoW models trained and evaluated successfully")

        # Indicate the end of the main process
        print("Main process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

# Ensure the script runs as the main program
if __name__ == "__main__":
    main()
