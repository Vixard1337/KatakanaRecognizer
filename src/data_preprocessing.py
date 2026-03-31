import os

def load_etl_files(data_path):
    # Check that the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path {data_path} does not exist.")
    files = os.listdir(data_path)
    return files
