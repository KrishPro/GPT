"""
Written by KrishPro @ KP

filename: `process.py`
"""

from tqdm import tqdm
from typing import Callable
from unidecode import unidecode
import multiprocessing as ml
import os

def count_lines(input_path:str, guessed_num_sentences:int=None):
    with open(input_path) as input_file:
        num_sentences = 0
        for _ in tqdm(input_file, total=guessed_num_sentences, desc="Counting sentences"):
            num_sentences += 1
    return num_sentences


def apply(input_path: str, output_path: str, function: Callable[[str], str], chunk_size: int = 2**15):
    pool = ml.Pool(processes=os.cpu_count())
    num_input_sentences = count_lines(input_path, guessed_num_sentences = 147877291)

    with open(input_path, 'r') as input_file:
        with open(output_path, 'w') as output_file:
            chunk = []
            for sentence in tqdm(input_file, total=num_input_sentences, desc=f"Applying {function.__name__}"):
                chunk.append(sentence)
                if len(chunk) == chunk_size:
                    chunk = pool.map(function, chunk)

                    output_file.write("".join(chunk))
                    chunk = []

            if len(chunk) > 0:
                chunk = pool.map(function, chunk)

                output_file.write("".join(chunk))
                chunk = []


if __name__ == "__main__":
    apply("<input-file>", "<output-file>", unidecode)
  


            