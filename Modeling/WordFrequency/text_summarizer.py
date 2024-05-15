import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import logging

# Set up logging
logging.basicConfig(
    filename='../../logs/text_summarizer_wf.log', 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class TextSummarizerWF:
    def __init__(self):
        """
        Initialize the TextSummarizer with a lemmatizer and stopwords set from NLTK.
        """
        logging.info('Initializing TextSummarizer...')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        logging.info('TextSummarizer initialized successfully.')

    def create_frequency_table(self, text_string):
        """
        Create a frequency table for words in a given text, excluding stopwords and using lemmatization.
        
        :param text_string: The text string from which to create the frequency table.
        :return: A dictionary with words as keys and their frequencies as values.
        """
        words = word_tokenize(text_string)
        freq_table = {}
        for word in words:
            word = self.lemmatizer.lemmatize(word)
            if word not in self.stop_words:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
        return freq_table

    def score_sentences(self, sentences, freq_table):
        """
        Score sentences by summing up the frequency of significant words they contain.
        
        :param sentences: List of sentences to score.
        :param freq_table: Frequency table of words.
        :return: A dictionary with sentence beginnings as keys and normalized scores as values.
        """
        sentence_value = {}
        for sentence in sentences:
            word_count_in_sentence_except_stop_words = 0
            words_in_sentence = word_tokenize(sentence.lower())
            sentence_key = sentence[:10]
            for word in words_in_sentence:
                if word in freq_table:
                    word_count_in_sentence_except_stop_words += 1
                    if sentence_key in sentence_value:
                        sentence_value[sentence_key] += freq_table[word]
                    else:
                        sentence_value[sentence_key] = freq_table[word]

            if word_count_in_sentence_except_stop_words > 0:
                sentence_value[sentence_key] /= word_count_in_sentence_except_stop_words
            else:
                sentence_value[sentence_key] = 0

        return sentence_value

    def find_average_score(self, sentence_value):
        """
        Find the average score of all scored sentences to set a cutoff for summary extraction.
        
        :param sentence_value: A dictionary of sentence scores.
        :return: The average score across all sentences.
        """
        sum_values = sum(sentence_value.values())
        average = sum_values / len(sentence_value) if sentence_value else 0
        return average

    def generate_summary(self, sentences, sentence_value, threshold):
        """
        Generate a summary by including sentences that meet a certain score threshold.
        
        :param sentences: List of all sentences.
        :param sentence_value: A dictionary of sentence scores.
        :param threshold: Score threshold.
        :return: A string that represents the summary.
        """
        summary = ''
        for sentence in sentences:
            sentence_key = sentence[:10]
            if sentence_key in sentence_value and sentence_value[sentence_key] >= threshold:
                summary += " " + sentence
        return summary

    def run_summarization(self, text):
        """
        Execute the summarization process on the provided text.
        
        :param text: The text to summarize.
        :return: The generated summary.
        """
        freq_table = self.create_frequency_table(text)
        sentences = sent_tokenize(text)
        sentence_scores = self.score_sentences(sentences, freq_table)
        threshold = self.find_average_score(sentence_scores) * 0.75
        summary = self.generate_summary(sentences, sentence_scores, threshold)
        return summary

# Example usage
# def main():
#     logging.info('Starting text summarization process.')
#     text = "Your text here..."
#     summarizer = TextSummarizer()
#     summary = summarizer.run_summarization(text)
#     logging.info(f'Generated summary: {summary}')
#     reference = "Your reference text here..."
#     scores = summarizer.calculate_rouge_bleu(reference, summary)
#     logging.info(f'Scores: {scores}')
#                                                                       
#   
# if __name__ == '__main__':
#     main()
