import math
import torch
import torch.nn.functional as F
from torch import nn

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len, dropout):
        d_head, remainder = divmod(d_model, num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(d_head)

        self.norm = nn.LayerNorm(d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # process key, query, values in one matmul
        self.kqv_proj = nn.Linear(d_model, 3 * d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.u_bias = nn.Parameter(torch.Tensor(num_heads, d_head))
        self.v_bias = nn.Parameter(torch.Tensor(num_heads, d_head))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        kqv = self.kqv_proj(self.norm(x))
        # kqv.shape == (batch_size, seq_len, 3 * d_model)
        key, query, value = torch.chunk(kqv, 3, dim=-1)
        # shape == (batch_size, seq_len, d_model)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)
    
        score = torch.mul(query, key) * self.scale
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        attn = F.softmax(score, -1)
        attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        out = torch.matmul(attn, value).transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.out_proj(out)
        out = self.out_dropout(x)

        return out
    
def skew(QEr):
    # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
    padded = F.pad(QEr, (1, 0))
    # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
    batch_size, num_heads, num_rows, num_cols = padded.shape
    reshaped = padded.view(batch_size, num_heads, num_cols, num_rows)
    # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
    Srel = reshaped[:, :, 1:, :].view_as(QEr)
    # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
    return Srel