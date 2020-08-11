import torch
import random
import zipfile

with zipfile.ZipFile('../../Dive-into-DL-PyTorch/data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

corpus_chars[:40]
