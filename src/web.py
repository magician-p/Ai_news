import json

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import FileResponse
from transformers import AutoTokenizer

from config import *
from model import ClassifyModel, SummaryModel
from predict import predict

app = FastAPI()

with open(PROCESSED_DATA_PATH['classify'] / 'id2label.json', 'r', encoding='utf-8') as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}
classify_model = ClassifyModel(PRETRAINED_MODEL_PATH['classify'], len(id2label))
classify_model.load_model(CLASSIFY_MODEL_PATH)

summary_model = SummaryModel(PRETRAINED_MODEL_PATH['summary'])
summary_model.load_model(SUMMARY_MODEL_PATH)

classify_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('classify'))
summary_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('summary'))

class RequestMessage(BaseModel):
    content: str

class ResponseMessage(BaseModel):
    category: str
    summary: str

@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / 'index.html')

@app.post('/summary')
def summary(request: RequestMessage):
    result = predict(request.content, id2label, classify_model, summary_model, classify_tokenizer, summary_tokenizer)
    result = ResponseMessage(
        category=result['label'],
        summary=result['summary'][0].replace('\n', '')
    )
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089)