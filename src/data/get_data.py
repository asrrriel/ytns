import os
import requests
import random

ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.94 Chrome/37.0.2062.94 Safari/537.36"

def download_images(source_file, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    with open(source_file, 'r') as f:
        for line in f:
            url = line.strip()
            try:
                file_name = url.split('#')[0].split('?')[0].split('/').pop()
                file_path = os.path.join(destination_dir, file_name)

                if os.path.exists(file_path):
                    print(f"File \"{file_path}\" already exists. Skipping download.")
                    continue

                response = requests.get(url, stream=True,headers={'User-Agent': ua})

                if(response.headers['Content-Type'].split('/')[0] != 'image'):
                    print(f"Bad media type: {response.headers['Content-Type']}")
                    continue

                if response.status_code != 200:
                    print(f"Error downloading {url}: {response.status_code}")
                    continue

                if file_name.split('.')[-1] == '':
                    print(file_name.split('.')[-1])

                with open(file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f"Downloaded \"{url}\" to \"{file_path}\"")

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
    download_images("sources/porn.txt","data/porn")
    download_images("sources/not_porn.txt","data/not_porn")