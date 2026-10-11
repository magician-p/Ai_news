import json

from datasets import ClassLabel, Features, Value, load_dataset
from transformers import AutoTokenizer

from config import (
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    pretrained_model_path,
)


def preprocess_data():
    dataset_dict = load_dataset(
        'csv',
        data_files={
            'train': str(RAW_DATA_PATH)
        },
        features=Features({
            'category': Value('string'),
            'text': Value('string'),
            'summary': Value('string')
        }),
        nrows=1000
    )

    dataset_dict = dataset_dict.filter(
        lambda batch: [
            text is not None and text.strip() != ""
            for text in batch['text']
        ],
        batched=True
    )

    classify_dataset = dataset_dict['train'].select_columns(['category', 'text'])
    summary_dataset = dataset_dict['train'].select_columns(['summary', 'text'])
    
    labels = sorted({
        label for label in classify_dataset['category']
    })
    with open(PROCESSED_DATA_PATH['classify'] / 'id2label.json', 'w', encoding='utf-8') as f:
        json.dump({
            idx: label for idx, label in enumerate(labels)
        }, f, indent=4, ensure_ascii=False)

    classify_dataset = classify_dataset.cast_column('category', ClassLabel(names=labels))

    classify_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('classify'))
    summary_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path('summary'))

    def classify_tokenize(batch):
        inputs = classify_tokenizer(batch['text'], truncation=True, max_length=MAX_INPUT_LENGTH)
        inputs['labels'] = batch['category']
        return inputs
    
    def summary_tokenize(batch):
        # max_length 截正文，max_target_length 截摘要；labels 由 tokenizer 自动生成
        return summary_tokenizer(
            text=batch['text'],
            text_target=batch['summary'],
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH
        )
    
    classify_dataset = classify_dataset.map(classify_tokenize, batched=True, remove_columns=['text', 'category'])
    summary_dataset = summary_dataset.map(summary_tokenize, batched=True, remove_columns=['text', 'summary'])

    classify_dataset_dict=classify_dataset.train_test_split(0.2, seed=42)
    summary_dataset_dict = summary_dataset.train_test_split(0.2, seed=42)

    classify_dataset_dict.save_to_disk(str(PROCESSED_DATA_PATH['classify']))
    summary_dataset_dict.save_to_disk(str(PROCESSED_DATA_PATH['summary']))


if __name__ == "__main__":
    preprocess_data()