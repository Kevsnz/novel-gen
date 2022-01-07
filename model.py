import math
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.modules.transformer as trans
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.ninp = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x * self.ninp + self.pe[:, : x.size(1)]
        return self.dropout(x)


'''class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm'''


'''class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        bs = q.size(0)

        # perform linear operation and split embeddings into h heads [bs, sl, h, dh]
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        d_k: float,
        mask: torch.Tensor = None,
        dropout=None,
    ):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # [bs, h, sl, sl]
        if mask is not None:
            mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output'''


'''class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x'''


'''class EncoderLayer(nn.Module):
    def __init__(self, d_embed, heads, nhid=2048, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_embed)
        self.norm_2 = nn.LayerNorm(d_embed)
        self.attn = nn.modules.activation.MultiheadAttention(d_embed, heads, dropout)
        self.ff = FeedForward(d_embed, nhid)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask):
        x2 = x.transpose(0, 1)
        x2 = self.attn(x2, x2, x2, attn_mask=mask)[0].transpose(0, 1)
        x = x + self.dropout_1(x2)
        x = self.norm_1(x)
        x = x + self.dropout_2(self.ff(x))
        x = self.norm_2(x)
        return x'''


'''class DecoderLayer(nn.Module):
    def __init__(self, d_embed, heads, nhid=2048, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_embed)
        self.norm_2 = nn.LayerNorm(d_embed)
        self.norm_3 = nn.LayerNorm(d_embed)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = nn.modules.activation.MultiheadAttention(d_embed, heads, dropout)
        self.attn_2 = nn.modules.activation.MultiheadAttention(d_embed, heads, dropout)
        self.ff = FeedForward(d_embed, nhid).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x).transpose(0, 1)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask)[0].transpose(0, 1))
        x2 = self.norm_2(x).transpose(0, 1)
        e_outputs = e_outputs.transpose(0, 1)
        x = x + self.dropout_2(
            self.attn_2(x2, e_outputs, e_outputs, src_mask)[0].transpose(0, 1)
        )
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x'''


'''class Encoder(nn.Module):
    def __init__(self, d_embed, N, heads, nhid=2048, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList(
            # [EncoderLayer(d_embed, heads, nhid) for _ in range(N)]
            [
                trans.TransformerEncoderLayer(
                    d_model=d_embed,
                    nhead=heads,
                    dim_feedforward=nhid,
                    dropout=dropout,
                    activation=F.gelu,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(N)
            ]
        )
        self.norm = norm.LayerNorm(d_embed)

    def forward(self, src: torch.Tensor, mask):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)'''

'''class Decoder(nn.Module):
    def __init__(self, d_embed, N, heads, nhid=2048, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList(
            # [DecoderLayer(d_model, heads, nhid) for _ in range(N)]
            [
                trans.TransformerDecoderLayer(
                    d_model=d_embed,
                    nhead=heads,
                    dim_feedforward=nhid,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(N)
            ]
        )
        self.norm = norm.LayerNorm(d_embed)

    def forward(self, trg: torch.Tensor, e_outputs, src_mask, trg_mask):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)'''


class Transformer(nn.Module):
    def __init__(self, src_vocab, d_embed, N, heads, nhid=2048, dropout=0.1):
        super().__init__()
        self.embed_src = nn.Embedding(src_vocab, d_embed)
        self.pe_src = PositionalEncoding(d_embed)
        # self.encoder = Encoder(d_embed, N, heads, nhid, dropout)

        self.encoder = trans.TransformerEncoder(
            trans.TransformerEncoderLayer(
                d_embed,
                heads,
                nhid,
                dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
            ),
            N,
            norm=nn.LayerNorm(d_embed),
        )

        self.out = nn.Linear(d_embed, src_vocab)
        self.mask = None
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.out.parameters():
            if p.dim() > 1:
                # init.xavier_uniform_(p)
                init.kaiming_uniform_(p, mode='fan_out')

    def forward(self, src: torch.Tensor, mask: bool = True):
        if mask:
            if self.mask is None or self.mask.size(1) != src.size(1):
                self.mask = trans.Transformer.generate_square_subsequent_mask(
                    src.size(1)
                ).to(src.device)
        else:
            self.mask = None

        src = self.embed_src(src)
        src = self.pe_src(src)
        e_outputs = self.encoder(src, self.mask)
        output = self.out(e_outputs)
        return output

    def load_model(filename: str, device: str):
        with open(filename, 'rb') as f:
            model = torch.load(f).to(device)
        print(f'Model loaded from {filename}')
        return model

    def save_model(self, epoch: int, dir: str):
        filename = f'model_{epoch}.pt'
        with open(os.path.join(dir, filename), 'wb') as f:
            torch.save(self, f)
            print(f'Model state saved to {filename}')
