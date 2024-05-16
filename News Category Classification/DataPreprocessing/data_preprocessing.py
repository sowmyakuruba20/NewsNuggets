import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
import os
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """
    A class for preprocessing textual data for NLP applications.

    Attributes:
        stop_words (set): A set of stopwords from the NLTK library for English.
        wordnet (WordNetLemmatizer): NLTK's lemmatizer.
        logger (Logger): Logger for logging messages and errors.
        output_dir (str): Directory where output files are saved if required.

    Methods:
        remove_urls(text): Removes URLs from a given text string.
        remove_punctuations(text): Removes punctuations from a given text string and handles new lines.
        to_lowercase(text): Converts a given text string to lowercase.
        tokenize_text(text): Tokenizes a given text string into words.
        remove_stopwords(tokens): Removes stopwords from a list of tokens.
        lemmatize_tokens(tokens): Applies lemmatization to a list of tokens.
        add_category_id_column(dataset, category_column): Adds a 'CategoryId' column to the dataset based on the factorization of the category column.
        split_train_test(dataset, text_column, category_column, test_size=0.2, random_state=None, save_to_csv=False, csv_dir=None): Splits the dataset into training and testing datasets.
        preprocess_text(dataset, text_column, category_column, test_size=0.2, random_state=None, save_to_csv=True): Full preprocessing pipeline from raw text to a split dataset ready for training/testing.
    """
    def __init__(self, output_dir):
        """
        Initializes the TextPreprocessor with a specified output directory for saving files.

        Parameters:
            output_dir (str): The directory where all output files should be saved.
        """
        self.stop_words = set(stopwords.words('english'))
        self.wordnet = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)  # Use the global logger
        self.output_dir = output_dir

    def remove_urls(self, text):
        """
        Removes URLs from the provided text string.

        Parameters:
            text (str): The text from which URLs will be removed.

        Returns:
            str: The text without URLs.
        """
        pattern = re.compile(r'http\S+|www\S+|https\S+')
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    def remove_punctuations(self, text):
        """
        Removes all punctuations from the provided text string and replaces new lines with spaces.

        Parameters:
            text (str): The text from which punctuations will be removed.

        Returns:
            str: The cleaned text without punctuations.
        """
        punctuation_pattern = r'[^\w\s]'
        cleaned_text = re.sub(punctuation_pattern, '', text)
        cleaned_text = cleaned_text.replace('\n', ' ')
        return cleaned_text

    def to_lowercase(self, text):
        """
        Converts all characters in the provided text string to lowercase.

        Parameters:
            text (str): The text to be converted.

        Returns:
            str: The lowercase text.
        """
        return text.lower()

    def tokenize_text(self, text):
        """
        Tokenizes the provided text string into individual words.

        Parameters:
            text (str): The text to be tokenized.

        Returns:
            list: A list of tokens (words).
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Removes stopwords from a list of tokens.

        Parameters:
            tokens (list): The list of word tokens from which stopwords will be removed.

        Returns:
            list: A list of tokens with stopwords removed.
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens):
        """
        Applies lemmatization to each token in the provided list of tokens.

        Parameters:
            tokens (list): The list of word tokens to be lemmatized.

        Returns:
            list: A list of lemmatized tokens.
        """
        return [self.wordnet.lemmatize(token) for token in tokens]

    def add_category_id_column(self, dataset, category_column):
        """
        Adds a 'CategoryId' column to the dataset by factorizing the specified category column.

        Parameters:
            dataset (DataFrame): The dataset to modify.
            category_column (str): The column name in the dataset from which category IDs will be generated.

        Returns:
            DataFrame: The dataset with a new 'CategoryId' column.
        """
        dataset['CategoryId'] = dataset[category_column].factorize()[0]
        return dataset

    def split_train_test(self, dataset, text_column, category_column, test_size=0.2, random_state=None, save_to_csv=False, csv_dir=None):
        """
        Splits the dataset into training and testing sets, and optionally saves them to CSV files.

        Parameters:
            dataset (DataFrame): The dataset to be split.
            text_column (str): The name of the column containing text data.
            category_column (str): The name of the column containing category labels.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int, optional): The seed used by the random number generator.
            save_to_csv (bool): If True, save the train and test datasets to CSV files.
            csv_dir (str, optional): Directory to save the CSV files if save_to_csv is True.

        Returns:
            tuple: Four elements (X_train, X_test, y_train, y_test), representing the split datasets.
        """
        X = dataset[text_column]
        y = dataset[category_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if save_to_csv and csv_dir:
            train_df = pd.DataFrame({text_column: X_train, category_column: y_train})
            test_df = pd.DataFrame({text_column: X_test, category_column: y_test})

            train_csv_path = os.path.join(csv_dir, 'pre_processed_train.csv')
            test_csv_path = os.path.join(csv_dir, 'pre_processed_test.csv')

            train_df.to_csv(train_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)

        return X_train, X_test, y_train, y_test
    
    def preprocess_text(self, dataset, text_column, category_column, test_size=0.2, random_state=None, save_to_csv=True):
        """
        Conducts a full preprocessing pipeline from raw text to a split dataset ready for training/testing.
        It processes text by removing URLs, punctuations, converting to lowercase, tokenizing, removing stopwords, and lemmatizing.
        Finally, it splits the dataset into training and testing datasets and optionally saves these splits to CSV files.

        Parameters:
            dataset (DataFrame): The dataset to preprocess.
            text_column (str): The column containing the text to preprocess.
            category_column (str): The column containing the category labels.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int, optional): The seed used by the random number generator for splitting.
            save_to_csv (bool): If True, the train and test data splits are saved to CSV files in the output directory.

        Returns:
            tuple: Four elements (X_train, X_test, y_train, y_test), representing the split datasets.
        """
        if dataset is None or dataset.empty:
            raise ValueError("Dataset is empty or None")

        dataset = self.add_category_id_column(dataset, category_column)

        dataset['Preprocessed_Text'] = dataset[text_column].apply(self.remove_urls)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.remove_punctuations)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.to_lowercase)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.tokenize_text)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.remove_stopwords)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.lemmatize_tokens)

        X_train, X_test, y_train, y_test = self.split_train_test(dataset, 'Preprocessed_Text', 'CategoryId', test_size=test_size, random_state=random_state, save_to_csv=save_to_csv, csv_dir=self.output_dir)

        return X_train, X_test, y_train, y_test
