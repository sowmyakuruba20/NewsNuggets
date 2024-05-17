import logging
import os
import numpy as np
import pandas as pd
import pickle  # Import pickle for saving models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
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

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
class ModelTrain:
    def __init__(self, output_dir, vectorizer_name):
        """
        Initializes the ModelTrain class with specified output directory and vectorizer name.
        
        Parameters:
            output_dir (str): The directory where all model outputs will be saved.
            vectorizer_name (str): The name of the vectorizer used for feature extraction to keep track of it in outputs.
        """
        self.model = None
        self.output_dir = output_dir
        self.vectorizer_name = vectorizer_name
        self.logger = logging.getLogger(__name__)
        self.results_data = []

    def train_model(self, model_name, vectorizer_name, x_train, y_train, x_test, y_test, search_spaces):
        """
        Trains a model with hyperparameter tuning using GridSearchCV. The best model is saved using pickle.

        Parameters:
            model_name (str): Short name for the model as specified in model_dict.
            x_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            x_test (array-like): Test data features.
            y_test (array-like): Test data labels.
            param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
        """
        if model_name in model_dict:
            self.logger.info(f"Training {model_name.upper()} model with hyperparameter tuning...")
            self.model = model_dict[model_name]
        else:
            self.logger.error(f"Model name {model_name} is not recognized.")
            return

        bayes_search = BayesSearchCV(self.model, search_spaces, n_iter=32, cv=5, n_jobs=-1)
        bayes_search.fit(x_train, y_train)

        self.logger.info("Training completed with Bayesian optimization.")
        self.logger.info(f"Best parameters for {model_name}: {bayes_search.best_params_}")
        best_model = bayes_search.best_estimator_

        # Save the best model using pickle
        model_filename = os.path.join(self.output_dir, f"{model_name}_best_model.pkl")
        with open(model_filename, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        # Evaluate and save results
        self.evaluate_model(model_name, vectorizer_name, best_model,  x_test, y_test, bayes_search.cv_results_)

    def evaluate_model(self, model_name, vectorizer_name,  model, x_test, y_test, cv_results):
        """
        Evaluates the model on the test data and records various metrics. Saves detailed results to a text file.

        Parameters:
            model_name (str): Name of the model being evaluated.
            model (sklearn.base.BaseEstimator): The trained model.
            x_test (array-like): Test data features.
            y_test (array-like): Test data labels.
            cv_results (dict): Results from GridSearchCV containing performance metrics and parameters.
        """
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Collect results for CSV
        self.results_data.append({
            'Model': model_name,
            'Vectorizer': self.vectorizer_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

        # Save detailed results to text file
        output_file_path = os.path.join(self.output_dir, f"{model_name}_results.txt")
        with open(output_file_path, 'a') as file:
            file.write(f"Evaluation results for {model_name}:\n")
            self.save_results(file, model_name, cv_results)
            file.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")
            report = classification_report(y_test, y_pred)
            file.write(report + "\n")
            confusion = confusion_matrix(y_test, y_pred)
            file.write(str(confusion) + "\n")

    def save_results(self, file, model_name, cv_results):
        """
        Writes the results of the hyperparameter tuning to a file.

        Parameters:
            file (file object): The file where results should be saved.
            model_name (str): Name of the model.
            cv_results (dict): Results from GridSearchCV.
        """
        file.write(f"Hyperparameter Tuning Results for {model_name}:\n")
        file.write("Parameters, Mean Test Score, Rank\n")
        for params, mean_score, rank in zip(cv_results['params'], cv_results['mean_test_score'], cv_results['rank_test_score']):
            file.write(f"{params}, {mean_score:.4f}, {rank}\n")

    def finalize_results_tfidf(self):
        """
        Compiles and saves all collected results into a CSV file.
        """
        df = pd.DataFrame(self.results_data)
        csv_path = os.path.join(self.output_dir, 'model_evaluation_summary1.csv')
        df.to_csv(csv_path, index=False)
    def finalize_results_bow(self):
        """
        Compiles and saves all collected results into a CSV file.
        """
        df = pd.DataFrame(self.results_data)
        csv_path = os.path.join(self.output_dir, 'model_evaluation_summary2.csv')
        df.to_csv(csv_path, index=False)
