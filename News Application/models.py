import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.wordnet = WordNetLemmatizer()

    def preprocess_text(self, text):
        text = self.remove_urls(text)
        text = self.remove_punctuations(text)
        text = self.to_lowercase(text)
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)

    def remove_urls(self, text):
        pattern = re.compile(r'http\S+|www\S+|https\S+')
        return re.sub(pattern, '', text)

    def remove_punctuations(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def to_lowercase(self, text):
        return text.lower()

    def tokenize_text(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens):
        return [self.wordnet.lemmatize(token) for token in tokens]

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def transform(self, texts):
        return self.vectorizer.transform(texts)

class TextSummarizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def create_frequency_table(self, text_string):
        words = word_tokenize(text_string)
        freq_table = {}
        for word in words:
            word = self.lemmatizer.lemmatize(word)
            if word in self.stop_words:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        return freq_table

    def score_sentences(self, sentences, freq_table):
        sentence_value = {}
        for sentence in sentences:
            word_count_in_sentence = len(word_tokenize(sentence))
            word_count_in_sentence_except_stop_words = 0
            for word_value in freq_table:
                if word_value in sentence.lower():
                    word_count_in_sentence_except_stop_words += 1
                    sentence_key = sentence[:10]
                    if sentence_key in sentence_value:
                        sentence_value[sentence_key] += freq_table[word_value]
                    else:
                        sentence_value[sentence_key] = freq_table[word_value]

            if word_count_in_sentence_except_stop_words > 0:
                sentence_value[sentence_key] /= word_count_in_sentence_except_stop_words
            else:
                sentence_value[sentence_key] = 0 

        return sentence_value

    def find_average_score(self, sentence_value):
        sum_values = sum(sentence_value.values())
        average = sum_values / len(sentence_value) if sentence_value else 0
        return average

    def generate_summary(self, sentences, sentence_value, threshold):
        sentence_count = 0
        summary = ''
        for sentence in sentences:
            if sentence[:10] in sentence_value and sentence_value[sentence[:10]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1
        return summary

    def run_summarization(self, text):
        freq_table = self.create_frequency_table(text)
        sentences = sent_tokenize(text)
        sentence_scores = self.score_sentences(sentences, freq_table)
        threshold = self.find_average_score(sentence_scores) * 1
        summary = self.generate_summary(sentences, sentence_scores, threshold)
        return summary
