import json

import torch
from transformers import AutoTokenizer

from config import *
from model import ClassifyModel, SummaryModel


class Predictor:
    def __init__(self, classify_model, summary_model, classify_tokenizer, summary_tokenizer, device, id2label):
        self.classify_model = classify_model.to(device)
        self.summary_model = summary_model.to(device)
        self.classify_tokenizer = classify_tokenizer
        self.summary_tokenizer = summary_tokenizer
        self.device = device
        self.id2label = id2label

    def predict(self, text):
        self.classify_model.eval()
        inputs = self.classify_tokenizer(text, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors='pt')
        with torch.no_grad():
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }
            pred = self.classify_model.predict(**inputs)
            label = self.id2label[int(pred.item())]
            return label

    def generate(self, text):
        self.summary_model.eval()
        inputs = self.summary_tokenizer(text, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors='pt')
        with torch.no_grad():
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }
            output = self.summary_model.generate(**inputs)
            return output

def predict(text, id2label, classify_model, summary_model, classify_tokenizer, summary_tokenizer):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = Predictor(
        classify_model, summary_model, classify_tokenizer, summary_tokenizer, device, id2label
    )

    label = predictor.predict(text)
    summary = predictor.generate(text)
    summary = summary_tokenizer.decode(summary, skip_special_tokens=True)
    return {
        'label': label,
        'summary': summary
    }

if __name__ == '__main__':
    with open(PROCESSED_DATA_PATH['classify'] / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    classify_model = ClassifyModel(PRETRAINED_MODEL_PATH['classify'], len(id2label))
    classify_model.load_model(CLASSIFY_MODEL_PATH)

    summary_model = SummaryModel(PRETRAINED_MODEL_PATH['summary'])
    summary_model.load_model(SUMMARY_MODEL_PATH)

    classify_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('classify'))
    summary_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('summary'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = Predictor(
        classify_model, summary_model, classify_tokenizer, summary_tokenizer, device, id2label
    )

    text = (
        "文华财经(编辑整理王欣)--据新加坡9月23日消息，"+
        "油价周四持稳在每桶75美元下方，先前政府公布的数据显示，"+
        "美国原油和汽油库存意外增加。尽管加拿大至美国最大的输油管线关闭长达一周，"+
        "但上周库存依旧上扬，确认了市场的看法，亦即油价今年剩余时间内将大多在石油输出国组织(OPEC)乐见的水准70-80美元间波动。"+
        "OPEC成员国利比亚的官员周三表示，OPEC在下月的会议上应会维持产出目标不变，并提高减产达成率。欢迎发表评论我要评论"
    )
    label = predictor.predict(text)
    print(f'文本分类结果：{label}')
    summary = predictor.generate(text)
    summary = summary_tokenizer.decode(summary, skip_special_tokens=True)[0].replace(' ', '')
    print(f'文本摘要结果：{summary}')
