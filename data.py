from typing import Protocol
import numpy as np
import tokenizers
from tokenizers.implementations import ByteLevelBPETokenizer


class Dataset(Protocol):
    trainset: np.ndarray
    evalset: np.ndarray
    testset: np.ndarray

    def dict_size(self) -> int:
        ...

    def tokenize(self, data: str) -> np.ndarray:
        ...

    def detokenize(self, data: np.ndarray) -> str:
        ...


class DatasetChar:
    def __init__(self) -> None:
        self.trainset: np.ndarray = None
        self.evalset: np.ndarray = None
        self.testset: np.ndarray = None
        self.char2num = {}
        self.num2char = []

    def dict_size(self) -> int:
        return len(self.char2num)

    def load_data(self, train_file: str, valid_fail: str, test_file: str):
        with open(train_file, 'r', encoding='utf8') as fp:
            self.trainset = self.tokenize(fp.read())

        with open(valid_fail, 'r', encoding='utf8') as fp:
            self.evalset = self.tokenize(fp.read())

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


class DatasetWordPart:
    trainset: np.ndarray
    evalset: np.ndarray
    testset: np.ndarray

    _txt2num: dict[str, int]
    _num2txt: list[str]

    def __init__(self, dict_file: str = None):
        self.trainset: np.ndarray = None
        self.evalset: np.ndarray = None
        self.testset: np.ndarray = None

        self._txt2num = {}
        self._num2txt = []
        self._load_dict(dict_file)

    def dict_size(self) -> int:
        return len(self._txt2num)

    def _load_dict(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            tokens = f.readlines()

        for t in tokens:
            if t.endswith('\n'):
                t = t[:-1]
            self._add(t.replace('\\n', '\n'))

        print(f'Dictionary loaded from \'{filename}\'')
        print(f'Dict len is {len(self._num2txt)}')

    def _add(self, c: str):
        if c in self._txt2num:
            return

        idx = len(self._num2txt)
        self._num2txt.append(c)
        self._txt2num[c] = idx

    def load_data(self, train_file: str, eval_file: str, test_file: str):
        print('Loading and tokenizing data...')
        with open(train_file, 'r', encoding='utf8') as fp:
            self.trainset = self.tokenize(fp.read())

        with open(eval_file, 'r', encoding='utf8') as fp:
            self.evalset = self.tokenize(fp.read())

        with open(test_file, 'r', encoding='utf8') as fp:
            self.testset = self.tokenize(fp.read())

        print('Data loaded!')
        print(
            f'Token count: train: {len(self.trainset)}, eval: {len(self.evalset)}, test: {len(self.testset)}'
        )

    def crop_data(self, train_count: int, eval_count: int, test_count: int):
        self.trainset = self.trainset[0:train_count]
        self.evalset = self.evalset[0:eval_count]
        self.testset = self.testset[0:test_count]
        print(
            f'Dataset cropped. Train len: {len(self.trainset)}, eval len: {len(self.evalset)}, test len: {len(self.testset)}'
        )

    def _get_ss_index(self, ss: str):
        while len(ss) > 0:
            w_idx = self._txt2num.get(ss)
            if w_idx is not None:
                return w_idx, len(ss)

            ss = ss[:-1]

        return None, None

    def tokenize(self, data: str) -> np.ndarray:
        idx = 0
        max_len = len(max(self._num2txt, key=len))

        data_tokens = []
        while idx < len(data):
            ss = data[idx : idx + max_len]
            ss_idx, l = self._get_ss_index(ss)
            if ss_idx is None:
                raise Exception('Cannot find token {ss}!')

            data_tokens.append(ss_idx)
            idx += l

        return np.array(data_tokens).astype(int)

    def detokenize(self, data_tokens: np.ndarray) -> str:
        data = [self._num2txt[t] for t in data_tokens]
        return ''.join(data)


class DatasetBPE:
    _tzr: ByteLevelBPETokenizer
    data: np.ndarray
    trainset: np.ndarray
    evalset: np.ndarray
    testset: np.ndarray

    def __init__(self, dict_file: str = None) -> None:
        self._tzr = tokenizers.Tokenizer.from_file(dict_file)
        print(f'DatasetBPE uses dictionary file \'{dict_file}\'')

    def load_data2(self, train_file: str, eval_file: str, token_limit: int = None):
        with open(train_file, 'r', encoding='utf8') as fp:
            data = fp.read()
        self.trainset = self.tokenize(data)
        if token_limit is not None:
            self.trainset = self.trainset[:token_limit]

        with open(eval_file, 'r', encoding='utf8') as fp:
            data = fp.read()
        self.evalset = self.tokenize(data)

        self.testset = []

        print('Data loaded!')
        print(
            f'Token count: train: {len(self.trainset)}, eval: {len(self.evalset)}, test: {len(self.testset)}'
        )

    def load_data(
        self,
        data_file: str,
        eval_part: float = 0.2,
        test_part: float = 0.1,
        token_limit=None,
    ):
        with open(data_file, 'r', encoding='utf8') as fp:
            data = fp.read()
        self.data = self.tokenize(data)
        if token_limit is not None:
            self.data = self.data[:token_limit]

        l = len(self.data)
        eval_len = max(2, int(round(l * eval_part)))
        test_len = max(2, int(round(l * test_part)))
        train_len = l - eval_len - test_len
        if train_len < 2:
            raise Exception('Not enough data')

        self.trainset = self.data[:train_len]
        self.evalset = self.data[train_len : train_len + eval_len]
        self.testset = self.data[train_len + eval_len :]

        print('Data loaded!')
        print(
            f'Token count: train: {len(self.trainset)}, eval: {len(self.evalset)}, test: {len(self.testset)}'
        )

    def crop_data(self, train_count: int, eval_count: int, test_count: int):
        self.trainset = self.trainset[0:train_count]
        self.evalset = self.evalset[0:eval_count]
        self.testset = self.testset[0:test_count]
        print(
            f'Dataset cropped. Train len: {len(self.trainset)}, eval len: {len(self.evalset)}, test len: {len(self.testset)}'
        )

    def tokenize(self, data: str) -> np.ndarray:
        encoded = self._tzr.encode(data)
        return np.array(encoded.ids)

    def detokenize(self, tokens: np.ndarray) -> str:
        return self._tzr.decode(tokens.tolist())

    def dict_size(self) -> int:
        return self._tzr.get_vocab_size()

    @property
    def blank_token(self) -> int:
        return self._tzr.token_to_id('<bl>')
