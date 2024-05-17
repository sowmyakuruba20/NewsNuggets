import re
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split 

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        """
        Initializes the TextPreprocessor object with required resources.
        """
        self.stop_words = set(stopwords.words('english'))
        self.wordnet = WordNetLemmatizer()
    
    
    def remove_urls(self, text):
        """
        Removes URLs from the input text.
        Args:
            text (str): The input text.
        Returns:
            str: Text with URLs removed.
        """
        pattern = re.compile(r'http\S+|www\S+|https\S+')
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    def remove_non_alphanumeric(self, text):
        """
        Removes non-alphanumeric characters from the input text.
        Args:
            text (str): The input text.
        Returns:
            str: Text with non-alphanumeric characters removed.
        """
        cleaned_text = ''.join(char if char.isalnum() else ' ' for char in text)
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    def to_lowercase(self, text):
        """
        Converts the input text to lowercase.
        Args:
            text (str): The input text.
        Returns:
            str: Lowercased text.
        """
        return text.lower()

    def remove_stopwords(self, text):
        """
        Removes stopwords from the input text.
        Args:
            text (str): The input text.
        Returns:
            str: Text with stopwords removed.
        """
        words = word_tokenize(text)
        filtered_words = [x for x in words if x not in self.stop_words]
        return ' '.join(filtered_words)

    def lemmatize_words(self, text):
        """
        Lemmatizes words in the input text.
        Args:
            text (str): The input text.
        Returns:
            str: Text with lemmatized words.
        """
        words = word_tokenize(text)
        lemmatized_words = [self.wordnet.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)


    def preprocess_text(self, dataset, text_column):
        """
        Preprocesses text data in the dataset.
        Args:
            dataset (pandas.DataFrame): The input dataset.
            text_column (str): The name of the column containing text data.
        Returns:
            pandas.DataFrame: Dataset with preprocessed text data.
        """
        logging.info("Starting text preprocessing")

        # Apply text preprocessing steps
        dataset['Preprocessed_Text'] = dataset[text_column].apply(self.remove_urls)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.remove_non_alphanumeric)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.to_lowercase)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.remove_stopwords)
        dataset['Preprocessed_Text'] = dataset['Preprocessed_Text'].apply(self.lemmatize_words)

        logging.info("Text preprocessing completed successfully")

        return dataset
    
   

