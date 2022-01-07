import math
import time as tm
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import Dataset, DatasetChar, DatasetBPE, DatasetWord, DatasetWordPart
from model import Transformer


# FILE_PATH = os.path.join('.', 'mydata_words')
FILE_PATH = os.path.join('..', 'data', 'novels')
FILE_DATA = os.path.join(FILE_PATH, 'novels.txt')
FILE_TRAIN = os.path.join(FILE_PATH, 'train.txt')
FILE_EVAL = os.path.join(FILE_PATH, 'valid.txt')
FILE_TEST = os.path.join(FILE_PATH, 'test.txt')
FILE_DICT = os.path.join(FILE_PATH, 'dictionary_byte_10000.json')
RESULT_DIR = os.path.join('models', 'bpe')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_INTER_MODELS = True
GEN_INTER_TEXTS = True
REP_INTER = True

SEQ_LEN = 256
HEADS = 16
EMBED = HEADS * 32
ENC_DEC_LAYERS = 6
N_HID = 512
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCH_BATCHES = 1
EVAL_BATCHES = 8
EPOCH_LIMIT = 200

LR = 0.001
LR_TGT = 0.0001
LR_DECAY = math.pow(LR_TGT / LR, 1 / (EPOCH_LIMIT - 1))  # 0.944
WEIGHT_DECAY = 0.02

REP_COUNT = 5
REP_INTERVAL = 5
rng: np.random.Generator = np.random.default_rng()
blank_token = None


def recalc_batch_params(ds: DatasetWordPart):
    global EPOCH_BATCHES, REP_INTERVAL
    token_count = len(ds.trainset)
    EPOCH_BATCHES = int((token_count / (SEQ_LEN * BATCH_SIZE)) // 100 * 100)
    if EPOCH_BATCHES == 0:
        EPOCH_BATCHES = int(token_count / (SEQ_LEN * BATCH_SIZE))

    REP_INTERVAL = int((EPOCH_BATCHES / REP_COUNT) // 100 * 100)
    if REP_INTERVAL == 0:
        REP_INTERVAL = int((EPOCH_BATCHES / REP_COUNT) // 20 * 20)
        if REP_INTERVAL == 0:
            REP_INTERVAL = int(EPOCH_BATCHES / REP_COUNT)


def split_to_batches(data: np.ndarray, bs) -> np.ndarray:
    cb = data.size // bs
    data = data[0 : cb * bs]
    data = data.reshape((bs, -1))
    return data


def get_train_sample_seq(
    batches: np.ndarray, seq_len: int, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_array = np.array(list(range(count)), dtype=int) * seq_len
    idx_array_seq = np.linspace(
        idx_array, idx_array + seq_len - 1, seq_len, dtype=int, axis=-1
    )

    # for idx_arr in idx_array_seq:
    #     for i1, i2 in zip(idx_arr[:-1], idx_arr[1:]):
    #         assert i2 == i1+1

    data: np.ndarray = np.take(batches, idx_array_seq, axis=1)
    target: np.ndarray = np.take(batches, idx_array_seq + 1, axis=1)

    data = torch.tensor(data).transpose(1, 0).contiguous()
    target = torch.tensor(target, dtype=torch.long).transpose(1, 0).contiguous()
    return data.to(device), target.to(device)


def get_train_sample_rnd(
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


def get_train_sample_rnd_no_shift(
    batches: np.ndarray, seq_len: int, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_start_range = batches.shape[1] - seq_len - 2
    idx_array: np.ndarray = rng.random(count) * idx_start_range
    idx_array = np.around(idx_array, 0).astype(int)
    idx_array_seq = np.linspace(
        idx_array, idx_array + seq_len - 1, seq_len, dtype=int, axis=-1
    )
    data: np.ndarray = np.take(batches, idx_array_seq, axis=1)
    target = torch.tensor(data, dtype=torch.long).transpose(1, 0).contiguous()

    data[-1, :] = blank_token
    data = torch.tensor(data).transpose(1, 0).contiguous()
    return data.to(device), target.to(device)


def create_model(src_vocab, trg_vocab):
    model = Transformer(
        src_vocab,
        trg_vocab,
        d_embed=EMBED,
        N=ENC_DEC_LAYERS,
        heads=HEADS,
        nhid=N_HID,
        dropout=DROPOUT,
    )
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         #nn.init.xavier_uniform_(p)
    #         nn.init.kaiming_normal_(p)
    print('New model created')
    return model.to(device)


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

        optimizer.zero_grad()

        loss = F.cross_entropy(pred.view(-1, vocab), tgt_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        int_loss += loss.item()
        total_loss += loss.item()
        if REP_INTER and (i + 1) % REP_INTERVAL == 0:
            lr = scheduler.get_last_lr()[0]
            dt = (tm.perf_counter() - start) / REP_INTERVAL
            print(
                f'    Batch {i+1:5}/{len(scr_data):5}, lr={lr:9.06f}: loss {int_loss/REP_INTERVAL:8.05f}, dt {dt:6.2f}s/batch'
            )
            int_loss = 0
            start = tm.perf_counter()

    return total_loss / len(scr_data)


def evaluate(model: nn.Module, eval_data: torch.Tensor, vocab: int):
    model.eval()
    eval_inputs, eval_targets = get_train_sample_rnd(eval_data, SEQ_LEN, EVAL_BATCHES)

    total_loss = 0.0
    with torch.no_grad():
        for input_batch, target_batch in zip(eval_inputs, eval_targets):
            output: torch.Tensor = model(input_batch)

            loss = F.cross_entropy(output.view(-1, vocab), target_batch.view(-1))
            total_loss += loss.item()

    return total_loss / len(eval_inputs)


def train(
    model: nn.Module,
    train_data: torch.Tensor,
    eval_data: torch.Tensor,
    ds: Dataset,
    gen_routine: callable,
):
    dt = datetime.datetime.now()
    dir = os.path.join(RESULT_DIR, f'run_{dt:%Y-%m-%d_%H-%M-%S}')
    os.makedirs(dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=LR_DECAY)
    epoch = 0

    try:
        print(
            f'Starting training. Epoch limit: {EPOCH_LIMIT}, batch count: {EPOCH_BATCHES}'
        )
        start_training = tm.perf_counter()
        for epoch in range(EPOCH_LIMIT):
            st_time = tm.monotonic()

            epoch_data, epoch_targets = get_train_sample_rnd(
                train_data, SEQ_LEN, EPOCH_BATCHES
            )

            loss = train_epoch(
                model, optimizer, scheduler, epoch_data, epoch_targets, ds.dict_size()
            )

            eval_loss = evaluate(model, eval_data, ds.dict_size())
            print(
                f'Epoch {epoch:3} done | '
                f'train loss {loss:8.05f}, '
                f'eval loss {eval_loss:8.05f} | '
                f'time {tm.monotonic()-st_time:5.1f}, '
                f'lr: {scheduler.get_last_lr()[0]:9.06f}'
            )

            if SAVE_INTER_MODELS:
                model.save_model(epoch, dir)

            if GEN_INTER_TEXTS:
                gen_text = gen_routine(model, 1000, ds)
                with open(
                    os.path.join(dir, f'gen_{epoch}.txt'), 'w', encoding='utf8'
                ) as f:
                    f.write(gen_text)
                print('Generated 1000 symbol text')

            scheduler.step()
            epoch += 1
        print(
            f'Training done! Epochs: {epoch}, total time: {tm.perf_counter()-start_training: .01f}s'
        )
        gen_text = gen_routine(model, 1000, ds)
        with open(
            os.path.join(dir, f'gen_{epoch}_final.txt'), 'w', encoding='utf8'
        ) as f:
            f.write(gen_text)
        print('Generated 1000 token text')
    except KeyboardInterrupt:
        print(
            f'Training interrupted! Epoch: {epoch}, total time: {tm.perf_counter()-start_training: .01f}s'
        )
    finally:


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
        model.save_model(epoch, dir)


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


def generate_wordpart(model, amount: int, ds: Dataset, primer: str = 'Привет, '):
    temp = 1
    model.eval()

    in_str = primer
    input_data = ds.tokenize(in_str)
    out_data = generate_tokens(model, amount, input_data, temp)

    out_str = ds.detokenize(out_data)
    return out_str


def generate_tokens(model, amount: int, input_data: np.ndarray, temp: float):
    out_data = torch.tensor(input_data)
    input_data = torch.tensor(input_data, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max(1, amount)):
            output_data = model(input_data, mask=False)
            output_data = output_data.squeeze(0)[-1]  # take last token

            # next_tokens = select_tokens_greedy(output_data)
            # next_tokens = select_tokens_temp(output_data, temp)
            next_tokens = select_tokens_topX_rnd(output_data, 500, temp)

            out_data = torch.cat([out_data, next_tokens.cpu()], 0)
            input_data = torch.cat([input_data, next_tokens.unsqueeze(0)], 1)
            input_data = input_data[:, -SEQ_LEN:]

    return out_data.int().numpy()


def select_tokens_topX_rnd(
    output_data: torch.Tensor, n: int = 2, temp: float = 1.0
) -> torch.Tensor:
    weights = F.softmax(output_data, -1)
    weights = weights.log().div(temp).exp()

    _, indices = torch.topk(weights, n, dim=0)
    idx = torch.multinomial(weights[indices], 1)
    return indices[idx].long()


def select_tokens_temp(output_data: torch.Tensor, temp: float) -> torch.Tensor:
    weights = F.softmax(output_data, -1)
    weights = weights.log().div(temp).exp()
    next_tokens = torch.multinomial(weights, 1).long()
    return next_tokens


def select_tokens_greedy(output_data: torch.Tensor) -> torch.Tensor:
    return torch.argmax(output_data, -1, keepdim=True)


def train_new_model(ds: Dataset, gen_routine: callable):
    vocab = ds.dict_size()
    train_batches = split_to_batches(ds.trainset, BATCH_SIZE)
    val_batches = split_to_batches(ds.evalset, BATCH_SIZE)
    test_batches = split_to_batches(ds.testset, BATCH_SIZE)

    model = create_model(vocab, vocab)
    #model = load_model('models\\bpe\\run_2022-01-02_12-05-30\\model_200.pt')

    train(model, train_batches, val_batches, ds, gen_routine)


def infer_model(ds: Dataset, model_path: str, gen_routine: callable):
    AMOUNT = 2000
    FILENAME = f'gennnnn.txt'
    model = Transformer.load_model(model_path, device)
    gibberish = gen_routine(
        model=model,
        amount=AMOUNT,
        ds=ds,
        # primer='Зима стояла снежная. Он ехал на машине через лес по заснеженной дороге. В машине было тепло, заряд держался на удивление хорошо. С такой эффективностью можно будет добраться не меньше чем до ближайшего большого города. Там будет и подзарядка, и ночлег, и потрясающий ужин.',
        primer='Быков вошел в рубку и сел в кресло капитана. Михаил Антонович посмотрел на него грустно. ',
    )
    # print(f'{gibberish}')
    with open(FILENAME, 'w', encoding='utf8') as f:
        f.write(gibberish)
    print(f'Generated {AMOUNT} tokens, stored to {FILENAME}')


def main_bpe():
    infer = True
    ds = DatasetBPE(FILE_DICT)
    global blank_token
    blank_token = ds.blank_token

    if infer:
        file = 'models\\bpe\\run_2022-01-02_14-48-25\\model_200.pt'
        infer_model(ds, file, generate_wordpart)
        return

    ds.load_data(FILE_DATA)
    # ds.crop_data(
    #     SEQ_LEN * BATCH_SIZE * 100 + BATCH_SIZE,
    #     len(ds.evalset),
    #     len(ds.testset),
    # )

    recalc_batch_params(ds)
    train_new_model(ds, generate_wordpart)


def main_wordpart():
    ds = DatasetWordPart(FILE_DICT)
    ds.load_data(FILE_TRAIN, FILE_EVAL, FILE_TEST)
    # ds.crop_data(
    #     SEQ_LEN * BATCH_SIZE * 100 + BATCH_SIZE,
    #     len(ds.evalset),
    #     len(ds.testset),
    # )
    recalc_batch_params(ds)

    train_new_model(ds, generate_wordpart)

    # file = 'models\\run_2021-12-18_08-22-15\\model_40.pt'
    # infer_model(ds, file, generate_wordpart)


def main_char():
    ds = DatasetChar()
    ds.load_data(FILE_TRAIN, FILE_EVAL, FILE_TEST)
    recalc_batch_params(ds)

    train_new_model(ds, generate_char)

    # model = load_model('models\\run_2021-11-30_18-15-21\\model_25.pt')
    # gibberish = generate(model, 20000, ds, 'Зима стояла снежная. Он ехал на машине через лес по заснеженной дороге. В машине было тепло, заряд держался на удивление хорошо. С такой эффективностью можно будет добраться не меньше чем до ближайшего большого города. Там будет и подзарядка, и ночлег, и потрясающий ужин.')
    # # print(f'{gibberish}')
    # with open(f'gennnnn.txt', 'w') as f:
    #     f.write(gibberish)


def main_word():
    ds = DatasetWord()
    ds.load_data(FILE_TRAIN, FILE_EVAL, FILE_TEST)
    recalc_batch_params(ds)

    # ds.clear_data()

    train_new_model(ds, generate_words)


def test():
    vocab_size = 100
    seq_len = 3
    batch_size = 4
    d_embed = 10

    data = np.arange(vocab_size)
    btchs = split_to_batches(data, bs=batch_size)  # sequences per batch

    src, tgt = get_train_sample_rnd(
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
    main_bpe()
    print('Done!')
