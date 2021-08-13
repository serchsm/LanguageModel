import requests
import zipfile
import numpy as np

from pathlib import Path
from abc import ABC

class PretrainedEmbeddings():
    def __init__(self, url, save_as):
        self.url = url
        self.file_path = save_as

    def download_file(self):
        if not self.file_path.is_file():
            print("Downloading embedding file...")
            file_stream = requests.get(self.url, allow_redirects=True, stream=True)
            with open(Path(self.file_path), mode="wb") as local_file:
                for chunk in file_stream:
                    local_file.write(chunk)
            print(f"Done")

    def parse_embedding_file(self):
        pass

    def build_embedding_matrix(self):
        pass


class GloveEmbeddings(PretrainedEmbeddings):
    def __init__(self, url, save_as, dimension):
        super().__init__(url, save_as)
        self.dimension = dimension
        self.dimension_to_file = {50: "glove.6B.50d.txt", 100: "glove.6B.100d.txt", 200: "glove.6B.200d.txt", 300: "glove.6B.300d.txt"}

    def extract_embedding_files(self):
        embedding_file = self.dimension_to_file[self.dimension]
        file_to_extract = self.file_path.parent / embedding_file
        with zipfile.ZipFile(self.file_path) as z:
            with open(file_to_extract, mode="wb") as fid:
                fid.write(z.read(embedding_file))
        return file_to_extract

    def get_embedding(self):
        self.download_file()
        path_to_embedding_file = self.extract_embedding_files()
        return path_to_embedding_file

    def parse_embedding_file(self, path_to_file):
        embedding_index = {}
        with open(path_to_file, mode='r') as fid:
            for line in fid:
                values = line.split()
                embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')
        return embedding_index

    def build_embedding_matrix(self, word_index):
        hits, misses = 0, 0
        file_path = self.get_embedding()
        embedding_index = self.parse_embedding_file(file_path)
        embedding_matrix = np.zeros((len(word_index), self.dimension))
        for word, index in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
                hits += 1
            else:
                misses += 1
        print(f"Embedding lookup hits: {hits}, misses: {misses}")
        return embedding_matrix

