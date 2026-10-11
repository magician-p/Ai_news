from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).parent.parent

# 数据集路径
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw" / "news.csv"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
PROCESSED_DATA_PATH = {
    'classify': PROCESSED_DATA_ROOT / "classify",
    'summary': PROCESSED_DATA_ROOT / "summary"
}

# 模型路径
MODEL_ROOT = PROJECT_ROOT / "models"
CLASSIFY_MODEL_PATH = MODEL_ROOT / "classify"
SUMMARY_MODEL_PATH = MODEL_ROOT / "summary"

# 预训练模型路径
PRETRAINED_MODEL_ROOT = PROJECT_ROOT / "pretrained"
BART_MODEL_PATH = PRETRAINED_MODEL_ROOT / "bart-base-chinese"
BERT_MODEL_PATH = PRETRAINED_MODEL_ROOT / "bert-base-chinese"
PRETRAINED_MODEL_PATH = {
    'classify': BERT_MODEL_PATH,
    'summary': BART_MODEL_PATH
}
PRETRAINED_MODEL_NAME = {
    'classify': 'google-bert/bert-base-chinese',
    'summary': 'OpenMOSS-Team/bart-base-chinese'
}
FILES_LIST = [
    'config.json',
    'model.safetensors',
    'tokenizer_config.json',
    'vocab.txt'
]
def pretrained_model_path(type:Literal['classify', 'summary']):
    local_path = PRETRAINED_MODEL_PATH[type]
    return (
        local_path
        if all( (local_path / file).exists() for file in FILES_LIST)
        else PRETRAINED_MODEL_NAME[type]
    )

# 日志路径
LOG_ROOT = PROJECT_ROOT / "logs"

# 检查点路径
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

# 序列长度（BERT / BART 位置编码上限均为 512，超出必须截断）
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64   # 摘要目标长度，新闻标题一般远短于 64

# 训练超参
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-5