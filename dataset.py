import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer


class JflegDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.X = self.dataset.input
        self.y = self.dataset.target
        self.source_len = 512
        self.target_len = 512
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ctext = str(self.X[index])
        ctext = ' '.join(ctext.split())

        text = str(self.y[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len,
                                                  pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.target_len,
                                                  pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long)
        }


def preprocess_jfleg():
    prefix = "jfleg"
    datasets = ["dev/dev", "test/test"]

    src, tgt = list(), list()

    with open("data/jfleg_source.txt", "w") as src_out, open("data/jfleg_target.txt", "w") as tgt_out:
        for dataset in datasets:
            f_src = open(os.path.join(prefix, f"{dataset}.src"), "r", encoding="utf-8")
            src_lines = f_src.readlines()

            refs = ["ref0", "ref1", "ref2", "ref3"]

            for ref in refs:
                f_tgt = open(os.path.join(prefix, f"{dataset}.{ref}"), "r", encoding="utf-8")
                tgt_lines = f_tgt.readlines()

                assert len(src_lines) == len(tgt_lines), "Source and Target should be parallel"

                for src, tgt in zip(src_lines, tgt_lines):
                    src_out.write(src)
                    tgt_out.write(tgt)


def process_and_save_data():
    with open('data/jfleg_target.txt') as f:
        target = f.readlines()

    with open('data/jfleg_source.txt') as f:
        source = f.readlines()

    inputs = ['grammar:' + i[:-1].replace(' .', '') for i in source]
    targets = [i[:-1].replace(' .', '') for i in target]

    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
    train_dataset = pd.DataFrame(columns=['input', 'target'])
    test_dataset = train_dataset.copy()

    train_dataset['input'] = X_train
    test_dataset['input'] = X_test
    train_dataset['target'] = y_train
    test_dataset['target'] = y_test

    train_dataset.to_csv('data/train.csv', index=False)
    test_dataset.to_csv('data/test.csv', index=False)


def prepare_data_for_training():
    train_data = JflegDataset('data/train.csv')
    test_data = JflegDataset('data/test.csv')

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)
    return train_loader, test_loader


if __name__=='__main__':
    process_and_save_data()