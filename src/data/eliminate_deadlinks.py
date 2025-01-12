import os
import requests
import random
from hashlib import sha256

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
                if response.status_code != 200:
                    print(f"Error downloading \"{url}\": {response.status_code}")
                    continue
                if response.headers['Content-Type'].split('/')[0] == 'image':
                    if url.split('://')[1].split('/')[0] == 'i.imgur.com':
                        hasher = sha256()
                        for chunk in response.iter_content(chunk_size=8192):
                            hasher.update(chunk)
                        if hasher.hexdigest() == '9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9':
                            print(f"\"{url}\" is a removed imgur image!")
                            continue
                    if url.split('://')[1].split('/')[0] == 'vidble.com':
                        hasher = sha256()
                        for chunk in response.iter_content(chunk_size=8192):
                            hasher.update(chunk)
                        if hasher.hexdigest() == '857c6c5db7854e52f7a2ddbbf80d288843dbff22e2c24fbe0456e71865542ac4':
                            print(f"\"{url}\" is a removed vidble image!")
                            continue
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