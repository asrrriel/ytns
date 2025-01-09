import os
import requests
import random

ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.94 Chrome/37.0.2062.94 Safari/537.36"

def no_deadlinks(source_file):
    with open(source_file, 'r') as f:
        lines = f.readlines()
    with open(source_file, 'w') as f:
        for line in lines:
            url = line.strip()
            print(f"Checking \"{url}\"")
            try:
                response = requests.get(url, stream=True,headers={'User-Agent': ua},timeout=5)
                response.raise_for_status()
                if response.headers['Content-Type'].split('/')[0] == 'image':
                    print(f"\"{url}\" is good!")
                    f.write(line)
                else:
                    print(f"Bad media type: {response.headers['Content-Type']}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading \"{url}\": {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

no_deadlinks("sources/porn.txt")
no_deadlinks("sources/not_porn.txt")