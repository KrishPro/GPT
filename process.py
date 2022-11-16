"""
Written by KrishPro @ KP

filename: `process.py`
"""

from tqdm import tqdm
from typing import Callable, List
from unidecode import unidecode
from tokenizers import Tokenizer
import multiprocessing as ml
import time
import math
import os

def count_lines(input_path:str, guessed_num_sentences:int=None):
    with open(input_path) as input_file:
        num_sentences = 0
        for _ in tqdm(input_file, total=guessed_num_sentences, desc="Counting sentences"):
            num_sentences += 1
    return num_sentences

def count_texts(input_path:str, guessed_num_sentences:int=None):
    with open(input_path) as input_file:
        num_texts = 0
        for t in tqdm(input_file, total=guessed_num_sentences, desc="Counting texts"):
            if t == '\n':
                num_texts += 1
        return num_texts


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

def get_texts(file):
    text = ""
    for t in file:
        if t == '\n':
            yield text.strip()
            text = ""
        else:
            text += t


def split_sequences(tokens: List[int], seq_len:int) -> List[List[int]]:
    if len(tokens) < (seq_len*0.5):
        seqs_tokens = []
    elif len(tokens) < (seq_len*1.5):
        tokens = tokens[:seq_len]
        tokens = tokens + ([0]*(seq_len-len(tokens)))
        seqs_tokens = [tokens]
    else:
        num_seqs_tokens = math.ceil(len(tokens)/seq_len)
        seqs_tokens = [tokens[(i*seq_len):((i+1)*seq_len)] for i in range(num_seqs_tokens)]
        if (seq_len*0.5) < len(seqs_tokens[-1]) < seq_len:
            seqs_tokens[-1] = seqs_tokens[-1] + ([0]*(seq_len-len(seqs_tokens[-1])))
        elif len(seqs_tokens[-1]) < (seq_len*0.5):
            seqs_tokens = seqs_tokens[:-1]

    return seqs_tokens

def process(input_path: str, output_path: str, vocab_path: str, chunk_size=2**10):

    tokenizer: Tokenizer = Tokenizer.from_file(vocab_path)
    num_texts = 7531765 if 7531765 else count_texts(input_path, guessed_num_sentences=147877291)
    seq_len = 512

    with open(input_path) as input_file:
        with open(output_path, 'w') as output_file:
            texts = []
            for i, text in enumerate(tqdm(get_texts(input_file), total=num_texts)):
                texts.append(text)
                if len(texts) == chunk_size:
                    seqs_tokens = [encoding.ids for encoding in tokenizer.encode_batch(texts)]

                    seqs_tokens = [seqs for tokens in seqs_tokens for seqs in split_sequences(tokens, seq_len=seq_len)]

                    if len(seqs_tokens) > 0 and len(seqs_tokens[0]) > 0:

                        output_file.write("\n".join([" ".join(map(str, tokens)) for tokens in seqs_tokens]) + ("" if i == (num_texts-1) else "\n"))
                    texts = []

def append_num_sentences(file_path: str, tmp_dir: str = '.'):
    num_sentences = count_lines(file_path, guessed_num_sentences=9085021)
    tmp_path = os.path.join(tmp_dir, str(time.time()))

    with open(file_path) as input_file:
        with open(tmp_path, 'w') as output_file:
            output_file.write(f"{num_sentences}\n")
            for t in tqdm(input_file, total=num_sentences):
                output_file.write(t)

    os.remove(file_path)
    os.rename(tmp_path, file_path)



if __name__ == "__main__":
    # apply("<input-file>", "<output-file>", unidecode)
    # process("<input-file>", "<output-file>", vocab_path='<vocab-path>')
    # append_num_sentences("<input-file>")
    pass