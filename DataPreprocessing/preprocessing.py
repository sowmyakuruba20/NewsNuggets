import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logging.basicConfig(
    filename='Logs/text_preprocessing.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class DataPreprocessor:
    def __init__(self):
        """
        Initialize the Text Preprocessor with necessary stopwords from NLTK and a lemmatizer.
        """
        logging.info('Initializing the Text Preprocessor...')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        logging.info('Text Preprocessor initialized successfully.')

    def preprocess(self, document):
        """
        Process a single document to remove stopwords, punctuation, special characters,
        extra spaces, convert to lowercase, and lemmatize the words.
        
        :param document: A string containing the text to be processed.
        :return: Tuple of cleaned and lemmatized document string and list of lemmatized words.
        """
        logging.info('Starting preprocessing of a document.')
        document = re.sub(r'[^a-z0-9\s]', '', document.lower())
        document = re.sub(r'\s+', ' ', document).strip()
        tokens = word_tokenize(document)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        cleaned_document = ' '.join(lemmatized_tokens)
        logging.info('Document preprocessing completed.')
        return cleaned_document, lemmatized_tokens

    def process_dataframe(self, df, column_name):
        """
        Process a specified column in DataFrame using the preprocess method and
        store results in 'cleaned_document' and 'word_lemmatized' columns.

        :param df: DataFrame containing the text data.
        :param column_name: The name of the column to process.
        :return: DataFrame with the processed text stored in the new columns.
        """
        logging.info(f'Processing DataFrame column: {column_name}')
        df[['cleaned_document', 'word_lemmatized']] = df[column_name].apply(
            lambda x: pd.Series(self.preprocess(x))
        )
        
        df['sentence_tokens'] = df[column_name].apply(sent_tokenize)
        logging.info('DataFrame processing completed.')
        return df
