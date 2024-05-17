import logging
from DataLoader.data_loader import DataLoader
from DataPreprocessing.data_preprocessing import TextPreprocessor
from TextVectorizer.vectorizer import Vectorizer
from Models.train_val import ModelTrain
from skopt.space import Real, Categorical, Integer
import pickle

# Set up logging configuration to capture all events
def setup_logging(log_file_path):
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main():
    # Setting up logging
    log_file_path = '/Users/madhu/OneDrive/Desktop/sentiment_classification/logs/application.log'
    setup_logging(log_file_path)
    logging.info("Main process started")

    # Path to the dataset
    data_path = "/Users/madhu/OneDrive/Desktop/sentiment_classification/Dataset/train.csv"
    data_loader = DataLoader(data_path)
    dataset = data_loader.load_data()
    logging.info("Data loaded successfully")

    # Initializing and using text preprocessor
    preprocessor = TextPreprocessor()
    dataset = preprocessor.preprocess_text(dataset, 'text')
    with open('text_preprocessor.pkl1', 'wb') as f:
      pickle.dump(dataset, f)
    logging.info("Data preprocessing completed")
    texts = dataset['Preprocessed_Text'].values
    labels = dataset['sentiment'].values
    
    #Define hyperparameter grids for model tuning
    '''param_grids = {
        'lr': {'penalty': ['l2'], 'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs'], 'max_iter': [100, 200], 'multi_class': ['ovr']},
        'dt': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'rf': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'ab': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},
        'nb': {}
    }'''
    search_spaces = {
        'lr': {'penalty': Categorical(['l2']), 'C': Real(0.1, 10, prior='log-uniform'), 'solver': Categorical(['liblinear', 'lbfgs']), 'max_iter': Integer(100, 200)},
        'dt': {'max_depth': Integer(10, 50), 'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(1, 4)},
        'rf': {'n_estimators': Integer(50, 200), 'max_depth': Integer(10, 50), 'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(1, 4)},
        'ab': {'n_estimators': Integer(50, 200), 'learning_rate': Real(0.1, 1, prior='log-uniform')},
        'nb': {'alpha': Real(1e-2, 1e+0, prior='log-uniform')}
        }

    # Initialize Vectorizer
    vectorizer = Vectorizer()
    X_train_tfidf, X_test_tfidf, y_train, y_test = vectorizer.vectorize_text(texts, labels, vectorizer_type='tfidf')
    output_dir_tfidf = '/Users/madhu/OneDrive/Desktop/sentiment_classification/Results'
    model_trainer_tfidf = ModelTrain(output_dir_tfidf, 'TFIDF')
    for model_name, search_spaces in search_spaces.items():
        model_trainer_tfidf.train_model(model_name, 'TFIDF', X_train_tfidf, y_train, X_test_tfidf, y_test, search_spaces)
        model_trainer_tfidf.finalize_results_tfidf()
        logging.info("TF-IDF models trained and evaluated successfully")
        
    vectorizer = Vectorizer()
    X_train_bow, X_test_bow, y_train, y_test = vectorizer.vectorize_text(texts, labels, vectorizer_type='bow')
    output_dir_tfidf = '/Users/madhu/OneDrive/Desktop/sentiment_classification/Results'
    model_trainer_tfidf = ModelTrain(output_dir_tfidf, 'BoW')
    for model_name, search_spaces in search_spaces.items():
        model_trainer_tfidf.train_model(model_name, 'TFIDF', X_train_bow, y_train, X_test_bow, y_test, search_spaces)
        model_trainer_tfidf.finalize_results_bow()
        logging.info("BoW models trained and evaluated successfully")
    
    

    # Indicate the end of the main process
    print("Main process completed successfully.")

if __name__ == "__main__":
    main()

