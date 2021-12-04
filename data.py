import torch
import numpy as np


class Dataset:
    def __init__(self) -> None:
        self.trainset: np.ndarray = None
        self.validset: np.ndarray = None
        self.testset: np.ndarray = None
        self.char2num = {}
        self.num2char = []

    def dict_size(self) -> int:
        return len(self.char2num)

    def load_data(self, train_file: str, valid_fail: str, test_file: str):
        with open(train_file, 'r', encoding='utf8') as fp:
            self.trainset = self.tokenize(fp.read())

        with open(valid_fail, 'r', encoding='utf8') as fp:
            self.validset = self.tokenize(fp.read())

        with open(test_file, 'r', encoding='utf8') as fp:
            self.testset = self.tokenize(fp.read())

        print('Data loaded!')
        print(f'Dict len is {len(self.char2num)}')

    def tokenize(self, data: str) -> np.ndarray:
        tokens = []
        for c in data:
            tokens.append(self._add_char(c))
        return np.array(tokens).astype(int)

    def detokenize(self, data: np.ndarray) -> str:
        out_chars = [self.num2char[i] for i in data]
        return ''.join(out_chars)

    def _add_char(self, s: str):
        if s not in self.char2num:
            self.char2num[s] = len(self.num2char)
            self.num2char.append(s)

        return self.char2num[s]


class DatasetWord:
    UNK_TOKEN = '<unk>'

    def __init__(self):
        self.trainset: np.ndarray = None
        self.validset: np.ndarray = None
        self.testset: np.ndarray = None
        self.word2num = {}
        self.num2word = []
        self.word2count = {}

    def dict_size(self) -> int:
        return len(self.word2num)

    def token_count(self) -> int:
        return sum(self.word2count.values())

    def _add_char(self, s: str):
        if s not in self.word2num:
            self.word2num[s] = len(self.num2word)
            self.num2word.append(s)
            self.word2count[s] = 0

        self.word2count[s] += 1
        return self.word2num[s]

    def tokenize(self, data: str) -> np.ndarray:
        tokens = []
        for c in data.split():
            if c:
                tokens.append(self._add_char(c.lower()))
        return np.array(tokens).astype(int)

    def detokenize(self, data: np.ndarray) -> str:
        out_chars = [self.num2word[i] for i in data]
        return ' '.join(out_chars)

    def load_data(self, train_file: str, valid_fail: str, test_file: str):
        with open(train_file, 'r', encoding='utf8') as fp:
            self.trainset = self.tokenize(fp.read())

        with open(valid_fail, 'r', encoding='utf8') as fp:
            self.validset = self.tokenize(fp.read())

        with open(test_file, 'r', encoding='utf8') as fp:
            self.testset = self.tokenize(fp.read())

        print(f'Data loaded: {sum(self.word2count.values())} words')
        print(f'Dict len is {len(self.word2num)}')

    def clear_data(self, min_occurrences: int = 2):
        tr_text = self.clear_set(self.trainset, min_occurrences)
        val_text = self.clear_set(self.validset, min_occurrences)
        tst_text = self.clear_set(self.testset, min_occurrences)

        self.word2num = {}
        self.num2word = []
        self.word2count = {}

        self.trainset = self.tokenize(tr_text)
        self.validset = self.tokenize(val_text)
        self.testset = self.tokenize(tst_text)

        print(f'Cleared data of {sum(self.word2count.values())} words')
        print(f'Dict len reduced to {len(self.word2num)}')

    def clear_set(self, set: np.ndarray, min_occurrences: int = 2) -> str:
        clean_text = []

        for token in set:
            w = self.num2word[token]
            if self.word2count[w] < min_occurrences:
                w = self.UNK_TOKEN

            clean_text.append(w)

        return ' '.join(clean_text)
