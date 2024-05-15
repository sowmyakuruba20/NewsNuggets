import pandas as pd
import numpy as np
import re
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine as cosine_distance
import networkx as nx

class TextSummarizerTR:
    def __init__(self):
        """
        Initialize the TextSummarizerTR class.

        Ensures the necessary nltk packages are downloaded.
        """
        # Ensure stopwords are downloaded
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def _normalize_whitespace(self, text):
        """
        Normalize the whitespace in the given text.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The text with normalized whitespace.
        """
        return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    def _sentence_similarity(self, sent1, sent2):
        """
        Calculate the similarity between two sentences.

        Args:
            sent1 (str): The first sentence.
            sent2 (str): The second sentence.

        Returns:
            float: The similarity score between the two sentences.
        """
        words1 = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sent1) if word.lower() not in self.stop_words]
        words2 = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sent2) if word.lower() not in self.stop_words]
        all_words = list(set(words1 + words2))
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]
        return 1 - cosine_distance(vector1, vector2)

    def run_summarization(self, text, top_n_sentences=5):
        """
        Summarize the given text.

        Args:
            text (str): The text to summarize.
            top_n_sentences (int, optional): The number of sentences to include in the summary. Defaults to 5.

        Returns:
            str: The summarized text.
        """
        sentences = sent_tokenize(text)
        n_sentences = len(sentences)
        sim_matrix = np.zeros((n_sentences, n_sentences))
        
        for i in range(n_sentences):
            for j in range(n_sentences):
                if i != j:
                    sim_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
        
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary = ' '.join([self._normalize_whitespace(s) for _, s in ranked_sentences[:top_n_sentences]])
        return summary
