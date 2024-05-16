import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    filename='Logs/data_loader.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class DataLoader:
    def __init__(self, dataset_folder):
        """
        Initialize the DataLoader object.

        Args:
            dataset_folder (str): The path to the folder where the dataset is located.

        Logs the start and completion of the initialization.
        """
        logging.info(f'Initializing DataLoader with dataset folder: {dataset_folder}')
        self.dataset_folder = dataset_folder
        logging.info('DataLoader initialization complete.')

    def load_data(self, file_name):
        """
        Load data from a file into a pandas DataFrame.

        Args:
            file_name (str): The name of the file to load.

        Returns:
            df (DataFrame): The loaded data as a pandas DataFrame. If the file cannot be loaded, an empty DataFrame is returned.

        Logs the start of the data loading process, and either a success message if the file is successfully loaded, an error message if the file is not found, or an error message with exception details if any other exception occurs.
        """
        logging.info(f'Starting to load data from {file_name}')
        file_path = os.path.join(self.dataset_folder, file_name)
        try:
            # Directly read CSV into DataFrame
            df = pd.read_csv(file_path)
            logging.info(f'Successfully loaded data from {file_name}')
            return df
        except FileNotFoundError:
            logging.error(f'File not found: {file_path}')
            return pd.DataFrame()
        except Exception as e:
            logging.error(f'Failed to load data from {file_name}: {e}')
            return pd.DataFrame()

# Example usage
# def main():
#     logging.info('Starting data loading process.')
#     dataset_folder = 'Dataset'
#     data_file = 'BBCnews.csv'
    
#     logging.info(f'Loading data from {dataset_folder}.')
#     data_loader = DataLoader(dataset_folder)
#     data = data_loader.load_data(data_file)

#     if data.empty:
#         logging.error('No data loaded. Check data source or path.')
#     else:
#         logging.info('Data loaded successfully.')
#         print(data.head())