import math
import time as tm
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import Dataset, DatasetWord
from model import Transformer


# FILE_PATH = os.path.join('.', 'mydata_words')
FILE_PATH = os.path.join('.','mydata')
FILE_TRAIN = os.path.join(FILE_PATH, 'train.txt')
FILE_VALID = os.path.join(FILE_PATH, 'valid.txt')
FILE_TEST = os.path.join(FILE_PATH, 'test.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 256
HEADS = 8
EMBED = HEADS * 8
ENC_DEC_LAYERS = 12
N_HID = 1024
DROPOUT = 0.1
SEQ_SHIFT = 1  # max(1,int(SEQ_LEN // 10))

#TOKEN_COUNT = 2971897  # words
TOKEN_COUNT =  24000000 # characters
BATCH_SIZE = 32
EPOCH_BATCHES = int((TOKEN_COUNT / (SEQ_LEN * BATCH_SIZE * 2)) // 100 * 100)
EVAL_BATCHES = 32
EPOCH_LIMIT = 40
LR = 0.0005
LR_TGT = 0.00005
LR_DECAY = math.pow(LR_TGT / LR, 1/EPOCH_LIMIT) #0.944
REP_INTERVAL = int(max(1, (EPOCH_BATCHES / 20) // 100) * 100)
rng: np.random.Generator = np.random.default_rng()


def split_to_batches(data: np.ndarray, bs) -> np.ndarray:
    cb = data.size // bs
    data = data[0 : cb * bs]
    data = data.reshape((bs, -1))
    return data


def get_train_sample(
    batches: np.ndarray, seq_len: int, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_start_range = batches.shape[1] - seq_len - 2
    idx_array: np.ndarray = rng.random(count) * idx_start_range
    idx_array = np.around(idx_array, 0).astype(int)
    idx_array_seq = np.linspace(
        idx_array, idx_array + seq_len - 1, seq_len, dtype=int, axis=-1
    )
    data: np.ndarray = np.take(batches, idx_array_seq, axis=1)
    target: np.ndarray = np.take(batches, idx_array_seq + 1, axis=1)

    data = torch.tensor(data).transpose(1, 0).contiguous()
    target = torch.tensor(target, dtype=torch.long).transpose(1, 0).contiguous()
    return data.to(device), target.to(device)


def create_model(src_vocab, trg_vocab):
    model = Transformer(
        src_vocab, trg_vocab, d_embed=EMBED, N=ENC_DEC_LAYERS, heads=HEADS, nhid=N_HID, dropout=DROPOUT
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)


def load_model(filename: str):
    with open(filename, 'rb') as f:
        model = torch.load(f).to(device)
    print(f'Model loaded from {filename}')
    return model


def save_model(model: nn.Module, epoch: int, dir: str):
    filename = f'model_{epoch}.pt'
    with open(os.path.join(dir, filename), 'wb') as f:
        torch.save(model, f)
        print(f'Model state saved to {filename}')


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.StepLR,
    scr_data: torch.Tensor,
    tgt_data: torch.Tensor,
    vocab: int,
):
    model.train()
    int_loss = 0
    total_loss = 0

    start = tm.perf_counter()
    for i, (scr_batch, tgt_batch) in enumerate(zip(scr_data, tgt_data)):
        pred: torch.Tensor = model(scr_batch)
        pred = pred.view(-1, vocab)

        optimizer.zero_grad()

        loss = F.cross_entropy(pred, tgt_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        int_loss += loss.item()
        if (i + 1) % REP_INTERVAL == 0:
            lr = scheduler.get_last_lr()[0]
            dt = (tm.perf_counter() - start)/REP_INTERVAL
            print(
                f'    Batch {i+1:5}/{len(scr_data):5}, lr={lr:9.06f}: loss {int_loss/REP_INTERVAL:8.05f}, dt {dt:6.2f}s/batch'
            )
            total_loss += int_loss
            int_loss = 0
            start = tm.perf_counter()

    return total_loss / len(scr_data)


def evaluate(model: nn.Module, eval_data: torch.Tensor, vocab: int):
    model.eval()
    eval_inputs, eval_targets = get_train_sample(eval_data, SEQ_LEN, EVAL_BATCHES)
    eval_targets = eval_targets.reshape(eval_targets.size(0), -1)

    total_loss = 0.0
    with torch.no_grad():
        for input_batch, target_batch in zip(eval_inputs, eval_targets):
            output = model(input_batch).view(-1, vocab)
            loss = F.cross_entropy(output, target_batch)
            total_loss += loss.item()

    return total_loss / len(eval_inputs)


def train(
    model: nn.Module, train_data: torch.Tensor, eval_data: torch.Tensor, ds: Dataset
):
    dt = datetime.datetime.now()
    dir = os.path.join('models', f'run_{dt:%Y-%m-%d_%H-%M-%S}')
    os.mkdir(dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=LR_DECAY)
    epoch = 0

    try:
        for epoch in range(EPOCH_LIMIT):
            st_time = tm.monotonic()

            epoch_data, epoch_targets = get_train_sample(
                train_data, SEQ_LEN, EPOCH_BATCHES
            )
            epoch_targets = epoch_targets.view(epoch_targets.size(0), -1)

            loss = train_epoch(
                model, optimizer, scheduler, epoch_data, epoch_targets, ds.dict_size()
            )

            eval_loss = evaluate(model, eval_data, ds.dict_size())
            print(
                f'Epoch {epoch:3} done: '
                f'train loss {loss:8.05f}, '
                f'eval loss {eval_loss:8.05f}, '
                f'time {tm.monotonic()-st_time:5.1f}'
            )

            save_model(model, epoch, dir)
            gen_text = generate_char(model, 1000, ds)
            with open(os.path.join(dir, f'get_{epoch}.txt'), 'w') as f:
                f.write(gen_text)
            print('Generated 1000 symbol text')

            scheduler.step()
            epoch += 1
    except KeyboardInterrupt:
        pass
    finally:
        save_model(model, epoch, dir)


def generate_char(model, amount: int, ds: Dataset, primer: str = 'Привет, '):
    temp = 0.25
    model.eval()

    in_str = primer
    add_len = SEQ_LEN - len(in_str)
    in_str = in_str.rjust(SEQ_LEN, ' ')
    input_data = ds.tokenize(in_str)

    out_data = generate_tokens(model, amount, input_data, temp)

    out_str = ds.detokenize(out_data)
    return out_str[add_len:]


def generate_words(
    model, amount: int, ds: Dataset, primer: str = 'привет , как дела ? '
):
    temp = 0.25
    model.eval()

    in_str = primer
    input_data = ds.tokenize(in_str)
    add_len = SEQ_LEN - len(input_data)
    input_data = np.concatenate((input_data, np.zeros(add_len)), axis=0)

    out_data = generate_tokens(model, amount, input_data, temp)

    out_str = ds.detokenize(out_data[add_len:])
    return out_str


def generate_tokens(model, amount: int, input_data: np.ndarray, temp: float):
    out_data = torch.tensor(input_data)
    input_data = torch.tensor(input_data, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max(1, amount)):
            output_data = model(input_data, mask=False).squeeze(0)
            # next_tokens = torch.argmax(output_data[-1], -1, keepdim=True)
            weights = F.softmax(output_data[-1], -1)
            weights = weights.log().div(temp).exp()
            next_tokens = torch.multinomial(weights, 1).long()
            if len(next_tokens.shape) > 1:
                next_tokens = next_tokens.squeeze(0)
            out_data = torch.cat([out_data, next_tokens.cpu()], 0)
            input_data = torch.cat([input_data, next_tokens.unsqueeze(0)], 1)
            input_data = input_data[:, -SEQ_LEN:]

    return out_data.int().numpy()


def train_new_model(ds: Dataset):
    vocab = ds.dict_size()
    train_batches = split_to_batches(ds.trainset, BATCH_SIZE)
    val_batches = split_to_batches(ds.validset, BATCH_SIZE)
    test_batches = split_to_batches(ds.testset, BATCH_SIZE)

    model = create_model(vocab, vocab)
    # model = load_model('models\\run_2021-12-03_22-36-21\\model_10.pt')
    train(model, train_batches, val_batches, ds)
    # print(generate_char(model, 1000, ds))


def main_char():
    ds = Dataset()
    ds.load_data(FILE_TRAIN, FILE_VALID, FILE_TEST)

    train_new_model(ds)

    # model = load_model('models\\run_2021-11-30_18-15-21\\model_25.pt')
    # gibberish = generate(model, 20000, ds, 'Зима стояла снежная. Он ехал на машине через лес по заснеженной дороге. В машине было тепло, заряд держался на удивление хорошо. С такой эффективностью можно будет добраться не меньше чем до ближайшего большого города. Там будет и подзарядка, и ночлег, и потрясающий ужин.')
    # # print(f'{gibberish}')
    # with open(f'gennnnn.txt', 'w') as f:
    #     f.write(gibberish)


def main_word():
    ds = DatasetWord()
    ds.load_data(FILE_TRAIN, FILE_VALID, FILE_TEST)

    # ds.clear_data()

    train_new_model(ds)


def test():
    vocab_size = 100
    seq_len = 3
    batch_size = 4
    d_embed = 10

    data = np.arange(vocab_size)
    btchs = split_to_batches(data, bs=batch_size)  # sequences per batch

    src, tgt = get_train_sample(
        btchs, seq_len=seq_len, count=2
    )  # sequence length, number of batches
    tgt = tgt.reshape(tgt.size(0), -1)
    print(f'Src:\n{src}')
    print(f'Tgt:\n{tgt}')

    model = create_model(100, 100, d_embed, 2, 2)
    print(f'Feeding batch, dims {src[0].shape}:\n{src[0]}')  # 4x3

    # embed = nn.Embedding(vocab_size, d_embed)  # 4x3 -> 4x3x10
    # pe = mod.PositionalEncoding(d_embed, dropout=0, max_len=50)  # 4x3x10 -> 4x3x10
    # norm = mod.Norm(d_embed)  # 4x3x10 -> 4x3x10

    # res = embed(src[0])
    # res = pe(res)
    # res = norm(res)
    res: torch.Tensor = model(src[0]).view(-1, vocab_size)
    print(f'Got dim {res.shape}:\n{res}')
    print(f'Target dim {tgt[0].shape}')
    l = F.cross_entropy(res, tgt[0])
    print(f'Loss: {l}')


if __name__ == '__main__':
    print('Welcome!')
    main_char()
    print('Done!')
