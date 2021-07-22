import requests
import zipfile
import numpy as np

from pathlib import Path
from abc import ABC

class PretrainedEmbeddings():
    def __init__(self, url, save_as):
        self.url = url
        # self.file_path = Path("./Embeddings/glove.6B.zip")
        self.file_path = save_as

    def download_file(self):
        if not self.file_path.is_file():
            file_stream = requests.get(self.url, allow_redirects=True, stream=True)
            with open(Path(self.file_path), mode="wb") as local_file:
                for chunk in file_stream:
                    local_file.write(chunk)

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



