"""
Written by KrishPro @ KP

filename: `vocab.py`
"""

import argparse
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def create_vocab(files: List[str], output_path: str):

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"])

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files=files, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-files', nargs='+', required=True)
    parser.add_argument('--output-path', required=True)

    args = parser.parse_args()
    create_vocab(files=args.input_files, output_path=args.output_path)