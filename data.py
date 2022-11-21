"""
Written by KrishPro @ KP

filename: `data.py`
"""

import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, file_path: str, chunk_size: int = 2**15) -> None:
        super().__init__()

        self.chunk_size = chunk_size
        self.file = open(file_path)

        self._reset()

    @staticmethod
    def parse_seq(seq): return list(map(int, seq.split(" ")))

    def _load_chunk(self):
        if not hasattr(self, 'previous_left'): self.previous_left = ""

        chunk = (self.previous_left + self.file.read(self.chunk_size)).split('\n')

        self.previous_left = chunk[-1]
        chunk = chunk[:-1]

        chunk = list(map(Dataset.parse_seq, chunk))

        return chunk

    def _reset(self):
        self.file.seek(0)
        self.len = int(next(self.file))
        self.current_start_idx = 0
        self.current_chunk = self._load_chunk()
        self.current_end_idx = len(self.current_chunk) - 1

    def load_chunk(self):
        self.current_start_idx += len(self.current_chunk)
        self.current_chunk = self._load_chunk()
        self.current_end_idx += len(self.current_chunk)

    def __getitem__(self, idx: int):
        if idx > self.current_end_idx:
            self.load_chunk()
            if self.current_chunk == []:
                self._reset()
                raise StopIteration
        return self.current_chunk[idx - self.current_start_idx]

    def __len__(self):
        return self.len
