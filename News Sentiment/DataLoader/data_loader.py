import logging
import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)  # Use the global logger

    def load_data(self):
        """Loads data from the CSV file specified at initialization."""
        try:
            dataset = pd.read_csv(self.file_path)
            self.logger.info("Data loaded successfully from %s", self.file_path)
            return dataset
        except FileNotFoundError:
            self.logger.error("The file %s was not found.", self.file_path)
            return None
        except Exception as e:
            self.logger.error("An error occurred while loading data from %s: %s", self.file_path, str(e))
            return None
