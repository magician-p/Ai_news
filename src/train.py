import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, logging

from config import *
from dataset import get_dataset
from model import ClassifyModel, SummaryModel

logging.disable_progress_bar()
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('classify'))
    collate_fn = DataCollatorWithPadding(tokenizer)
    train_dataset = get_dataset('classify', 'train')
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True
    )
    
    with open(PROCESSED_DATA_PATH['classify'] / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    
    model = ClassifyModel(
        pretrained_model_path('classify'), 
        num_labels=len(id2label))
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scalar = torch.amp.GradScaler(device.type, enabled=True)
    
    loss_ = float('inf')
    
    for epoch in tqdm(range(1, EPOCHS+1), desc='Total Training...', leave=True, position=0, ncols=100):
        loss_e = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False, position=1, ncols=70):
            
            with torch.autocast(device.type, dtype=torch.float16, enabled=True):
                inputs = {k: v.to(device) for k, v in batch.items()}
                output = model(**inputs)
                loss = output['loss']
            
            optim.zero_grad()
            scalar.scale(loss).backward()
            scalar.step(optim)
            scalar.update()

            loss_e += loss.item()
        if loss_e/len(train_dataloader) < loss_:
            loss_ = loss_e
            model.save_model(CLASSIFY_MODEL_PATH)   


if __name__ == "__main__":
    train()