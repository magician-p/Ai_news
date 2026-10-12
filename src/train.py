import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    logging,
)

from config import *
from dataset import get_dataset
from model import ClassifyModel, SummaryModel

logging.disable_progress_bar()
@dataclass
class Trainconfig:
    # dataclass 只把「带类型注解」的类属性识别为字段，
    # 没有注解的只是普通类属性，不会出现在生成的 __init__ 签名里
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    save_steps: int = 50
    judge_metric: Literal['loss', 'f1', 'recall'] = 'f1'
    early_stop_patience: int = 3
    save_path: Path | None = None
    checkpoint_path: Path | None = None
    log_dir: Path | None = None

class Trainer:
    def __init__(self, model, train_dataset, collate_fn, device, trainconfig, val_dataset: Any | None = None):
        self.model = model.to(device)
        self.train_data_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size = trainconfig.batch_size, shuffle=True)
        self.val_data_loader: DataLoader | None = None
        if val_dataset is not None:
            self.val_data_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=trainconfig.batch_size)
        with open(PROCESSED_DATA_PATH['classify'] / 'id2label.json', 'r', encoding='utf-8') as f:
            self.id2label = json.load(f)
        self.optim = torch.optim.Adam(self.model.parameters(), trainconfig.lr)
        self.step = 1
        self.device = device
        self.trainconfig = trainconfig
        self.scalar = torch.amp.GradScaler(device.type, enabled=True)
        self.writer = SummaryWriter(trainconfig.log_dir / time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.best_loss = -float('inf')
        self.early_stop_count = 0

    
    def train(self):
        self._load_checkpoint(self.trainconfig.checkpoint_path)
        for epoch in tqdm(range(1, self.trainconfig.epochs+1), desc='Total Training...', leave=True, position=0, ncols=100):
            self.model.train()
            for batch in tqdm(self.train_data_loader, desc=f'Epoch {epoch}/{self.trainconfig.epochs}', leave=False, position=1, ncols=80):
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=True):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    output = self.model(**inputs)
                    loss = output['loss']
                
                self.optim.zero_grad()
                self.scalar.scale(loss).backward()
                self.scalar.step(self.optim)
                self.scalar.update()

                if self.step % self.trainconfig.save_steps == 0:
                    tqdm.write(f'Step {self.step}: Loss {loss.item()}')
                    self.writer.add_scalar('Loss', loss.item(), self.step)
                    self._save_checkpoint()
                    
                    cur_metric = -loss.item()
                    if self.val_data_loader is not None:
                        cur_metric = self.val()[self.trainconfig.judge_metric]
                    if cur_metric > self.best_loss:
                        self.best_loss = cur_metric
                        self.early_stop_count = 0
                        self.model.save_model(self.trainconfig.save_path)
                        tqdm.write(f'Saved model to {self.trainconfig.save_path}')
                    else:
                        self.early_stop_count += 1
                        if self.early_stop_count >= self.trainconfig.early_stop_patience:
                            tqdm.write(f'Early stop at step {self.step}')
                            break
                self.step += 1
                    
    def val(self):
        val_loader = self.val_data_loader
        if val_loader is None:
            return {}
        self.model.eval()
        loss_sum = 0.0
        pred_list = []
        labels_list = []
        for batch in tqdm(val_loader, desc='Validation', leave=False, position=1, ncols=80):
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**inputs)
                loss = output['loss']
                logits = output['logits']
                labels = inputs['labels']

                loss_sum += loss.item()
                pred_list.append(torch.argmax(logits, dim=-1).reshape(-1)) # (B, L, C) -> (B, L) -> (B*L)
                labels_list.append(labels.reshape(-1)) # (B, L) -> (B*L)
        loss_sum /= len(val_loader)
                
        preds = torch.cat(pred_list, dim=-1).cpu().numpy()
        targets = torch.cat(labels_list, dim=-1).cpu().numpy()
        return {'loss': loss_sum, **self._compute_metrics(preds, targets)}

    def _compute_metrics(self, preds, labels):
        # -100 是 seq2seq 的 label padding 位，不能参与指标计算
        mask = labels != -100
        preds, labels = preds[mask], labels[mask]
        return {
            'f1': f1_score(labels, preds, average='weighted', zero_division=0),  # pyright: ignore[reportArgumentType]
            'recall': recall_score(labels, preds, average='weighted', zero_division=0)  # pyright: ignore[reportArgumentType]
        }

    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scalar_state_dict': self.scalar.state_dict(),
            'step': self.step,
            'best_loss': self.best_loss,
            'early_stop_count': self.early_stop_count
        }
        torch.save(checkpoint, self.trainconfig.checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scalar.load_state_dict(checkpoint['scalar_state_dict'])
            self.step = checkpoint['step']
            self.best_loss = checkpoint['best_loss']
            self.early_stop_count = checkpoint['early_stop_count']
            tqdm.write(f'Loaded checkpoint from {checkpoint_path}')
        else:
            tqdm.write(f'No checkpoint found at {checkpoint_path}')


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(PROCESSED_DATA_PATH['classify'] / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    
    model = ClassifyModel(PRETRAINED_MODEL_PATH['classify'], len(id2label))
    train_dataset = get_dataset('classify', 'train')
    val_dataset = get_dataset('classify', 'test')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH['classify'])
    collate_fn = DataCollatorWithPadding(tokenizer, return_tensors='pt')
    trainconfig = Trainconfig(
        save_path = CLASSIFY_MODEL_PATH,
        checkpoint_path = CHECKPOINT_ROOT / 'classify' / 'checkpoint.pt',
        log_dir = LOG_ROOT / 'classify'
    )
    trainer = Trainer(
        model,
        train_dataset,
        collate_fn,
        device,
        trainconfig,
        val_dataset
    )
    trainer.train()

    model = SummaryModel(PRETRAINED_MODEL_PATH['summary'])
    train_dataset = get_dataset('summary', 'train')
    val_dataset = get_dataset('summary', 'test')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH['summary'])
    collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model.bart, padding=True)
    trainconfig = Trainconfig(
        save_path = SUMMARY_MODEL_PATH,
        checkpoint_path = CHECKPOINT_ROOT / 'summary' / 'checkpoint.pt',
        log_dir = LOG_ROOT / 'summary'
    )
    trainer = Trainer(
        model,
        train_dataset,
        collate_fn,
        device,
        trainconfig,
        val_dataset
    )
    trainer.train()