import unittest
from src.data_preprocessing import load_etl_files

class TestDataPreprocessing(unittest.TestCase):
    def test_load_etl_files(self):
        # Specify the path to ETL1 data
        etl1_path = "data/raw/ETL1"
        files = load_etl_files(etl1_path)
        self.assertTrue(len(files) > 0, "No files found in ETL1 directory")
        print("ETL1 files loaded successfully:", files)

if __name__ == "__main__":
    unittest.main()
