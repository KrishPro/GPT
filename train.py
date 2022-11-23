"""
Written by KrishPro @ KP

filename: `train.py`
"""

try:
    from model import LanguageModel
    from data import Dataset
except ImportError:
    from gpt.model import LanguageModel
    from gpt.data import Dataset

import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "dims": {
        "d_model":128, 
        "n_heads":2, 
        "dim_feedforward":128,
        "vocab_size":30_000,
        "n_layers":1,
        "pad_idx":0,
        "max_len":512,
        "dropout_p":0.1
    },

    "data_path": ".data/processed.txt",
    "label_smoothing": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 3,
    "epochs": 1000
}

def generate_log(epoch_idx: int, batch_idx: int, total_epochs: int, total_batches: int, loss: torch.Tensor):
    global prev_frame_time
    
    # Calculating fps
    prev_frame_time = time.time() if 'prev_frame_time' not in globals() else prev_frame_time
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # Calculating eta
    eta = (total_batches - batch_idx) / fps
    mm, ss = divmod(eta, 60)
    hh, mm = divmod(mm, 60)

    log = f"{epoch_idx:05d}/{total_epochs} |  {batch_idx:05d}/{total_batches} | ({batch_idx/total_batches*100:.3f}%) | [{fps:.3f}it/s] | eta: {int(hh):02d}:{int(mm):02d}:{int(ss):02d} | Loss: {loss:.5f}"
    return log


def train_step(batch: torch.Tensor, model: LanguageModel, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, config: dict, val_step=False, optimizer_step=True):
        
        with torch.set_grad_enabled(not val_step):
            batch: torch.Tensor = batch.to(device)

            predictions: torch.Tensor = model(batch[:, :-1])

            loss: torch.Tensor = criterion(predictions.view(-1, config['dims']['vocab_size']), batch[:, 1:].reshape(-1))

            if val_step:
                 return loss.detach()
            
            loss.backward()

            if optimizer_step:
                optimizer.step()
                optimizer.zero_grad()

            return loss.detach()

def train(config):

    ## Setting up dataset
    dataset = Dataset(config['data_path'])
    dataloader = data.DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, collate_fn=torch.tensor)
    len_dataloader = len(dataloader.batch_sampler)

    ## Setting up model
    model = LanguageModel(**config["dims"]).to(device)

    ## Setting up optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=config['dims']['pad_idx'], label_smoothing=config['label_smoothing'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    try:
        for epoch_idx in range(config['epochs']):
            for batch_idx, batch in enumerate(dataloader):

                loss = train_step(batch, model, criterion, optimizer, config, optimizer_step=batch_idx % 4 == 0)
                
                print(generate_log(epoch_idx, batch_idx, config['epochs'], len_dataloader, loss=loss), end='\r')
    except KeyboardInterrupt:
        print()
        print(generate_log(epoch_idx, batch_idx, config['epochs'], len_dataloader, loss=loss))
         

if __name__ == '__main__':
     train(config)