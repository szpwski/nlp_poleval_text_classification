import requests
import zipfile
import io
import os

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

# Specify URLs with data
urls = ["http://2019.poleval.pl/task6/task_6-1.zip", "http://2019.poleval.pl/task6/task6_test.zip"]

# Replace the 'extract_path' with the folder where you want to extract the contents
extract_path = './data/'

# Call the function to download and extract the zip file with training data
download_and_extract_zip(urls[0], extract_path)

# Call the function to download and extract the zip file with test data
download_and_extract_zip(urls[1], extract_path, specific_folder='Task6/task 01')
