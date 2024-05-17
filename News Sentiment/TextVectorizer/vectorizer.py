import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from scipy.sparse import csr_matrix
import pickle

class Vectorizer:
    def __init__(self):
        """
        Initializes the Vectorizer object.
        """
        pass
    def vectorize_text(self, texts, labels, vectorizer_type='tfidf', test_size=0.2, random_state=42):
        """
        Vectorizes the input texts and splits them into training and testing sets.
        Args:
            texts (list): A list of text data.
            labels (list): A list of corresponding labels.
            vectorizer_type (str): Type of vectorizer to use ('tfidf' or 'bow').
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.
        Returns:
            tuple: A tuple containing x_train, x_test, y_train, and y_test.
                x_train : Features for training.
                x_test : Features for testing.
                y_train : Labels for training.
                y_test : Labels for testing.
        """
        logging.info("Vectorizing with {}...".format(vectorizer_type))
        
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer()
        elif vectorizer_type == 'bow':
            vectorizer = CountVectorizer()
        else:
            raise ValueError("Invalid vectorizer_type. Choose 'tfidf' or 'bow'.")
        
        x = vectorizer.fit_transform(texts)
        with open('tfidf_vectorizer.pkl1', 'wb') as f:
            pickle.dump(x, f)
        
        label_counts = pd.Series(labels).value_counts()
        count_1 = label_counts.get(1, 0)
        count_0 = label_counts.get(0, 0)
        
        if count_1 < count_0:
            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x, labels)
        elif count_0 < count_1:
            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x, labels)
        else:
            # Dataset is balanced, no oversampling needed
            x_resampled, y_resampled = x, labels

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled)

        return x_train, x_test, y_train, y_test


