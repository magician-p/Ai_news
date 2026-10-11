import torch
from torch import nn
from transformers import AutoModel


class ClassifyModel(nn.Module):
    def __init__(self, model_path, num_labels:int) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def save_model(self, save_path):
        self.bert.save_pretrained(save_path)
        torch.save(self.fc.state_dict(), save_path / 'fc.pt')
    
    def load_model(self, load_path):
        self.bert = AutoModel.from_pretrained(load_path)
        self.fc = torch.load(load_path / 'fc.pt')

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        logits = self.fc(outputs.pooler_output) # (B, H) -> (B, C)
        loss = self.loss_fn(logits, labels)
        return {
            'loss': loss, 
            'logits': logits,
        }
    
    def predict(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        logits = self.fc(outputs.pooler_output) # (B, H) -> (B, C)
        return torch.argmax(logits, dim=-1)

class SummaryModel(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.bart = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(self.bart.config.hidden_size, self.bart.config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def save_model(self, save_path):
        self.bart.save_pretrained(save_path)
        torch.save(self.fc.state_dict(), save_path / 'fc.pt')
    
    def load_model(self, load_path):
        self.bart = AutoModel.from_pretrained(load_path)
        self.fc = torch.load(load_path / 'fc.pt')

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs = self.bart(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logits = self.fc(outputs.last_hidden_state) # (B, T, H) -> (B, T, V)
        loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        return {
            'loss': loss,
            'logits': logits,
        }

    def generate(self, input_ids, token_type_ids, attention_mask):
        memery = self.bart.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        decoder_input_ids = torch.tensor([self.bart.config.decoder_start_token_id] * input_ids.shape[0]).to(input_ids.device) # (B, 1)
        decoder_hidden_states = memery.last_hidden_state
        is_finished = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)
        while True:
            outputs = self.bart.decoder(
                input_ids=decoder_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=decoder_hidden_states,
                encoder_attention_mask=attention_mask
            )
            logits = self.fc(outputs.last_hidden_state) # (B, T, H) -> (B, T, V)
            pred_ids = torch.argmax(logits[:, -1, :], dim=-1) # (B, V) -> (B, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, pred_ids], dim=-1)
            decoder_hidden_states = outputs.last_hidden_state
            is_finished = is_finished | (pred_ids.squeeze(-1) == self.bart.config.eos_token_id)
            if is_finished.all():
                break

        return decoder_input_ids