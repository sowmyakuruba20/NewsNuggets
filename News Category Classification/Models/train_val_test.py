import logging
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Mapping of model short names to their respective classes
model_dict = {
    'lr': LogisticRegression(),
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(),
    'ab': AdaBoostClassifier(algorithm='SAMME'),
    'nb': MultinomialNB()
}

class ModelTrain:
    """
    A class for training and evaluating machine learning models using Bayesian optimization.

    Attributes:
        model (sklearn estimator): The current machine learning model being used.
        output_dir (str): Directory where model outputs like serialized models and evaluation results are saved.
        vectorizer_name (str): Name of the vectorizer used for preprocessing text data, used for tracking in outputs.
        logger (Logger): Logger object for logging information during the model's training and evaluation process.
        results_data (list of dicts): Stores performance metrics of models for later summarization.

    Methods:
        train_model(model_name, x_train, y_train, x_test, y_test, search_spaces): Trains a model using Bayesian optimization.
        evaluate_model(model_name, model, x_test, y_test, cv_results): Evaluates a trained model and logs the results.
        finalize_results(): Saves the collected model performance metrics to a CSV file.
    """

    def __init__(self, output_dir, vectorizer_name):
        """
        Initializes the ModelTrain class with an output directory and vectorizer name.

        Parameters:
            output_dir (str): The directory where all model outputs will be saved.
            vectorizer_name (str): The name of the vectorizer used to preprocess text, which helps track the setup in outputs.
        """
        self.model = None
        self.output_dir = output_dir
        self.vectorizer_name = vectorizer_name
        self.logger = logging.getLogger(__name__)
        self.results_data = []

    def train_model(self, model_name, x_train, y_train, x_test, y_test, search_spaces):
        """
        Trains a model specified by `model_name` using Bayesian optimization over specified `search_spaces`.

        Parameters:
            model_name (str): The short name of the model to be trained, as defined in the global `model_dict`.
            x_train (np.array): Training data features.
            y_train (np.array): Training data labels.
            x_test (np.array): Testing data features.
            y_test (np.array): Testing data labels.
            search_spaces (dict): The search spaces for hyperparameter optimization defined per model.

        Returns:
            None, but the best model is saved to a file, and its performance is evaluated and logged.
        """
        if model_name in model_dict:
            self.logger.info(f"Training {model_name.upper()} model with Bayesian optimization...")
            self.model = model_dict[model_name]
        else:
            self.logger.error(f"Model name {model_name} is not recognized.")
            return

        bayes_search = BayesSearchCV(self.model, search_spaces, n_iter=32, cv=5, n_jobs=-1)
        bayes_search.fit(x_train, y_train)

        self.logger.info("Training completed with Bayesian optimization.")
        self.logger.info(f"Best parameters for {model_name}: {bayes_search.best_params_}")
        best_model = bayes_search.best_estimator_

        # Save the best model
        model_filename = os.path.join(self.output_dir, f"{model_name}_best_model.pkl")
        with open(model_filename, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        self.evaluate_model(model_name, best_model, x_test, y_test, bayes_search.cv_results_)

    def evaluate_model(self, model_name, model, x_test, y_test, cv_results):
        """
        Evaluates the trained model on the test dataset and logs various performance metrics.

        Parameters:
            model_name (str): The name of the model being evaluated.
            model (sklearn.base.BaseEstimator): The trained model.
            x_test (np.array): Testing data features.
            y_test (np.array): Testing data labels.
            cv_results (dict): Results from Bayesian optimization, including best parameters and scores.

        Returns:
            None, but writes the evaluation results to a text file and appends them to `results_data`.
        """
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        self.results_data.append({
            'Model': model_name,
            'Vectorizer': self.vectorizer_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

        output_file_path = os.path.join(self.output_dir, f"{model_name}_results.txt")
        with open(output_file_path, 'a') as file:
            file.write(f"Evaluation results for {model_name}:\n")
            file.write(f"Best Parameters: {cv_results['params'][cv_results['rank_test_score'].argmin()]}\n")
            file.write(f"Validation Mean Score: {cv_results['mean_test_score'][cv_results['rank_test_score'].argmin()]:.4f}\n")
            file.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")
            report = classification_report(y_test, y_pred)
            file.write(report + "\n")
            confusion = confusion_matrix(y_test, y_pred)
            file.write(str(confusion) + "\n")

    def finalize_results(self):
        """
        Compiles all collected model performance metrics and saves them into a CSV file.

        Returns:
            None, but a CSV file is created at `output_dir` containing the summary of model evaluations.
        """
        df = pd.DataFrame(self.results_data)
        csv_path = os.path.join(self.output_dir, 'model_evaluation_summary.csv')
        df.to_csv(csv_path, index=False)
