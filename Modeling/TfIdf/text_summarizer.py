import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import math

class TextSummarizerTFIDF:
    def __init__(self):
        """
        Initialize the TextSummarizerTFIDF class.
        """
        self.stopWords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def read_text_from_csv(self, file_path, text_column):
        """
        Read text from a CSV file.

        Args:
            file_path (str): The path to the CSV file.
            text_column (str): The column containing the text.

        Returns:
            Series: The text column from the CSV file.
        """
        df = pd.read_csv(file_path)
        return df[text_column].dropna()

    def create_frequency_table(self, text_string):
        """
        Create a frequency table for the words in the text.

        Args:
            text_string (str): The text.

        Returns:
            dict: The frequency table.
        """
        words = word_tokenize(text_string)
        freqTable = {}
        for word in words:
            word = self.lemmatizer.lemmatize(word.lower())
            if word not in self.stopWords:
                freqTable[word] = freqTable.get(word, 0) + 1
        return freqTable

    def create_frequency_matrix(self, sentences):
        """
        Create a frequency matrix for the sentences.

        Args:
            sentences (list): The sentences.

        Returns:
            dict: The frequency matrix.
        """
        frequency_matrix = {}
        for sent in sentences:
            freq_table = {}
            words = word_tokenize(sent)
            for word in words:
                word = self.lemmatizer.lemmatize(word.lower())
                if word not in self.stopWords:
                    freq_table[word] = freq_table.get(word, 0) + 1
            frequency_matrix[sent[:15]] = freq_table
        return frequency_matrix

    def create_tf_matrix(self, freq_matrix):
        """
        Create a term frequency (TF) matrix from the frequency matrix.

        Args:
            freq_matrix (dict): The frequency matrix.

        Returns:
            dict: The TF matrix.
        """
        tf_matrix = {}
        for sent, f_table in freq_matrix.items():
            tf_table = {}
            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence
            tf_matrix[sent] = tf_table
        return tf_matrix

    def create_documents_per_words(self, freq_matrix):
        """
        Create a table of the number of documents per word.

        Args:
            freq_matrix (dict): The frequency matrix.

        Returns:
            dict: The table of the number of documents per word.
        """
        word_per_doc_table = {}
        for sent, f_table in freq_matrix.items():
            for word in f_table:
                word_per_doc_table[word] = word_per_doc_table.get(word, 0) + 1
        return word_per_doc_table

    def create_idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
        """
        Create an inverse document frequency (IDF) matrix.

        Args:
            freq_matrix (dict): The frequency matrix.
            count_doc_per_words (dict): The table of the number of documents per word.
            total_documents (int): The total number of documents.

        Returns:
            dict: The IDF matrix.
        """
        idf_matrix = {}
        for sent, f_table in freq_matrix.items():
            idf_table = {}
            for word in f_table:
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
            idf_matrix[sent] = idf_table
        return idf_matrix

    def create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        """
        Create a term frequency-inverse document frequency (TF-IDF) matrix.

        Args:
            tf_matrix (dict): The TF matrix.
            idf_matrix (dict): The IDF matrix.

        Returns:
            dict: The TF-IDF matrix.
        """
        tf_idf_matrix = {}
        for (sent, f_table) in tf_matrix.items():
            tf_idf_table = {}
            for word in f_table:
                tf_idf_table[word] = f_table[word] * idf_matrix[sent][word]
            tf_idf_matrix[sent] = tf_idf_table
        return tf_idf_matrix

    def score_sentences(self, tf_idf_matrix):
        """
        Score the sentences based on their TF-IDF scores.

        Args:
            tf_idf_matrix (dict): The TF-IDF matrix.

        Returns:
            dict: The scores of the sentences.
        """
        sentenceValue = {}
        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = sum(f_table.values())
            sentenceValue[sent] = total_score_per_sentence / len(f_table) if len(f_table) > 0 else 0
        return sentenceValue

    def find_average_score(self, sentenceValue):
        """
        Find the average score of the sentences.

        Args:
            sentenceValue (dict): The scores of the sentences.

        Returns:
            float: The average score.
        """
        average = sum(sentenceValue.values()) / len(sentenceValue) if len(sentenceValue) > 0 else 0
        return average

    def generate_summary(self, sentences, sentenceValue, threshold):
        """
        Generate a summary of the text.

        Args:
            sentences (list): The sentences.
            sentenceValue (dict): The scores of the sentences.
            threshold (float): The threshold for including a sentence in the summary.

        Returns:
            str: The summary.
        """
        summary = ''
        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= threshold:
                summary += " " + sentence
        return summary.strip()

    def run_summarization(self, text):
        """
        Run the summarization process on the text.

        Args:
            text (str): The text.

        Returns:
            str: The summary.
        """
        sentences = sent_tokenize(text)
        total_documents = len(sentences)

        freq_matrix = self.create_frequency_matrix(sentences)
        tf_matrix = self.create_tf_matrix(freq_matrix)
        count_doc_per_words = self.create_documents_per_words(freq_matrix)
        idf_matrix = self.create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        tf_idf_matrix = self.create_tf_idf_matrix(tf_matrix, idf_matrix)

        sentenceValue = self.score_sentences(tf_idf_matrix)

        threshold = self.find_average_score(sentenceValue)

        summary = self.generate_summary(sentences, sentenceValue, 0.6 * threshold)

        return summary