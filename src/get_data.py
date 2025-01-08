import os
import requests

def download_images(source_file, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    with open(source_file, 'r') as f:
        for line in f:
            url = line.strip()
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                file_name = os.path.basename(url)
                file_path = os.path.join(destination_dir, file_name)

                with open(file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f"Downloaded \"{url}\"")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

def make_data_dirs():
    """Creates the data directory and subdirectoriúes for porn and not_porn."""
    data_dir = "data"
    porn_dir = os.path.join(data_dir, "porn")
    not_porn_dir = os.path.join(data_dir, "not_porn")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(porn_dir):
        os.makedirs(porn_dir)

    if not os.path.exists(not_porn_dir):
        os.makedirs(not_porn_dir)

    print("Directories created or already exist.")

if __name__ == "__main__":
    make_data_dirs()
    download_images("sources/porn.txt", "data/porn")
    download_images("sources/not_porn.txt", "data/not_porn")