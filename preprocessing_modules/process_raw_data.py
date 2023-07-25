import requests
import zipfile
import io
import os
import pandas as pd

def download_and_extract_zip(url, extract_path, file_type="txt", specific_folder=None):
    '''
    Function downloads and extracts ZIP folder from URL to specified path

    Args:
        url - URL of the ZIP folder
        extract_path - path where to extract ZIP file
        file_type - type of files to be extracted from ZIP file (default is "txt")
        specific_folder - specific folder in the ZIP file to be extracted if exists
    '''

    # Send an HTTP GET request to download the zip file
    response = requests.get(url)
    
    if response.status_code == 200:
        # Read the content of the downloaded zip file
        zip_content = io.BytesIO(response.content)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_content, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                # Check if specific_folder is None or the file_name starts with the specific_folder
                if specific_folder is None or file_name.startswith(specific_folder):
                    # Check if the file has the desired file_type extension
                    if file_name.endswith(file_type):
                        # Get the base name of the file (without subfolders)
                        file_base_name = os.path.basename(file_name)

                        # Create the destination folder if it doesn't exist
                        os.makedirs(extract_path, exist_ok=True)
                        
                        # Extract the file to the extract_path (without subfolders)
                        with zip_ref.open(file_name) as source, open(os.path.join(extract_path, file_base_name), "wb") as target:
                            target.write(source.read())

        print(f"Extraction of {url} completed successfully.")
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}")

def read_text_files_into_dataframe(data_folder : str):
    """
    Function reads text files from specified data folder and saves them as dataframe

    Args:
        data_folder - path to folder with text data
    """

    texts = []  # List to store the text data
    labels = []  # List to store the corresponding labels
    split = [] # List to store information about split

    # For each file in the data folder
    for file_name in os.listdir(data_folder):

        # Specify whether it is training or test part
        split_name = 'train' if file_name.startswith('training') else 'test'

        file_path = os.path.join(data_folder, file_name)

        # Read from text file
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_name.endswith('text.txt'):
                read_texts = file.read().split("\n")
                texts = texts + read_texts
                split = split + [split_name] * len(read_texts)

            elif file_name.endswith('tags.txt'):
                labels = labels + file.read().split("\n")
            
            else:
                raise Exception("Not implemented error!")
            
       
    # Create a pandas DataFrame
    df = pd.DataFrame({'text': texts[:-1], 'label': labels[:-1], 'split': split[:-1]})
    
    return df
