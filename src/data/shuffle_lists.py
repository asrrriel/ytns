import os
import random

def shuffle(list_path):
    with open(list_path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(list_path, 'w') as f:
        f.writelines(lines)

    print(f"Shuffled {list_path}")

shuffle("sources/porn.txt")
shuffle("sources/not_porn.txt")