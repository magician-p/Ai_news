import torch
from torch import nn
from transformers import AutoModel
from transformers.models.bart.modeling_bart import shift_tokens_right


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
        fc_state_dict = torch.load(load_path / 'fc.pt')
        self.fc.load_state_dict(fc_state_dict)

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
        fc_state_dict = torch.load(load_path / 'fc.pt')
        self.fc.load_state_dict(fc_state_dict)

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None, decoder_input_ids=None):
        # BART 不接受 token_type_ids，因此这里不再往下传
        # BartModel 没有 labels 参数，若不显式给 decoder_input_ids，它会拿 input_ids 右移当解码输入，
        # 导致解码长度=正文长度(512) 与 labels 长度(摘要长度) 对不上，必须自己由 labels 右移构造
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.bart.config.pad_token_id, self.bart.config.decoder_start_token_id
            )
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        logits = self.fc(outputs.last_hidden_state) # (B, T, H) -> (B, T, V)
        loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        return {
            'loss': loss,
            'logits': logits,
        }

    def generate(self, input_ids, attention_mask, token_type_ids = None, max_length=64):
        memery = self.bart.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        decoder_input_ids = torch.full(
            (input_ids.shape[0], 1),
            self.bart.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device
        ) # (B, 1)
        is_finished = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)
        while True:
            outputs = self.bart.decoder(
                input_ids=decoder_input_ids,
                attention_mask=None,
                encoder_hidden_states=memery.last_hidden_state,
                encoder_attention_mask=attention_mask
            )
            logits = self.fc(outputs.last_hidden_state) # (B, T, H) -> (B, T, V)
            pred_ids = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1) # (B, V) -> (B)
            decoder_input_ids = torch.cat([decoder_input_ids, pred_ids], dim=-1)
            pred_ids = pred_ids.masked_fill(is_finished, self.bart.config.pad_token_id)
            is_finished = is_finished | (pred_ids.squeeze(-1) == self.bart.config.eos_token_id)
            if is_finished.all():
                break

        return decoder_input_ids