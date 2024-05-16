# Description: Main file for running the news summarization
from DataLoader.data_loader import DataLoader  
from DataPreprocessing.preprocessing import DataPreprocessor  
from Modeling.WordFrequency.text_summarizer import TextSummarizerWF  
from Modeling.TfIdf.text_summarizer import TextSummarizerTFIDF
from Modeling.TextRank.text_summarizer import TextSummarizerTR
from Evaluation.evaluate import TextEvaluator 

import pandas as pd
import logging
import os

logging.basicConfig(
    filename='Logs/application.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class NewsSummarizer:
    def __init__(self, dataset_folder, data_file):
        """
        Initialize the NewsSummarizer class.

        Args:
            dataset_folder (str): The folder where the dataset is located.
            data_file (str): The name of the data file.
        """
        self.dataset_folder = dataset_folder
        self.data_file = data_file
        self.data_loader = DataLoader(dataset_folder)
        self.preprocessor = DataPreprocessor()
        self.summarizer_wf = TextSummarizerWF()
        self.summarizer_tfidf = TextSummarizerTFIDF()
        self.summarizer_tr = TextSummarizerTR()
        self.evaluate = TextEvaluator()
        self.score_columns = [
            'rouge_1_f', 'rouge_1_precision', 'rouge_1_recall',
            'rouge_2_f', 'rouge_2_precision', 'rouge_2_recall',
            'rouge_l_f', 'rouge_l_precision', 'rouge_l_recall',
            'bleu_score'
        ]

    def load_data(self):
        """
        Load the data from the data file.

        Returns:
            DataFrame: The loaded data.
        """
        data = self.data_loader.load_data(self.data_file)
        if data.empty:
            logging.error('No data loaded. Check data source or path.')
        else:
            logging.info('Data loaded successfully.')
            print(data.head())
        return data

    def preprocess_data(self, data):
        """
        Preprocess the loaded data.

        Args:
            data (DataFrame): The loaded data.

        Returns:
            DataFrame: The preprocessed data.
        """
        if not data.empty:
            processed_df = self.preprocessor.process_dataframe(data, 'document')
            logging.info('Data preprocessing completed.')
            output_file_path = os.path.join(self.dataset_folder, self.data_file)
            processed_df.to_csv(output_file_path, index=False)
            logging.info(f'Successfully saved processed data to {output_file_path}')
        else:
            logging.error('No data loaded. Check data source or path.')
        return processed_df

    def summarize_and_evaluate(self, df, text_column):
        """
        Summarize and evaluate the preprocessed data.

        Args:
            df (DataFrame): The preprocessed data.
            text_column (str): The name of the column containing the text to be summarized.

        Returns:
            None
        """
        if text_column in df.columns:
            print("Summarization started")
            for summarizer, name in [(self.summarizer_wf, 'wf'), (self.summarizer_tfidf, 'tfidf'), (self.summarizer_tr, 'tr')]:
                logging.info(f'Summarization and evaluation started for {name}.')
                df[f'generated_summary_{name}'] = df[text_column].apply(summarizer.run_summarization)
                df_scores = df.apply(lambda row: self.evaluate.calculate_rouge_bleu(row['summary'], row[f'generated_summary_{name}']), axis=1, result_type='expand')
                df_wf = pd.concat([df, df_scores], axis=1)
                grouped_average_scores = df_wf.groupby('category')[self.score_columns].mean()
                output_file_path = os.path.join('Results', f'average_scores_by_category_{name}.csv')
                grouped_average_scores.to_csv(output_file_path, index=False)
                model_file_path = os.path.join('Results', f'news_summarizer_{name}.pkl')
                self.evaluate.save_model(summarizer, model_file_path)
                logging.info(f'Summarization and evaluation completed for {name}.')
            logging.info('Summarization completed')
            print("Summarization completed")
        else:
            logging.error(f"Column '{text_column}' does not exist in the DataFrame.")

    def run(self):
        """
        Run the news summarizer.

        Returns:
            None
        """
        logging.info('Starting data loading process.')
        data = self.load_data()
        processed_df = self.preprocess_data(data)
        self.summarize_and_evaluate(processed_df, 'document')

if __name__ == '__main__':
    """
    Main entry point of the application.
    """
    summarizer = NewsSummarizer('Dataset', 'Processed_BBCnews.csv')
    summarizer.run()