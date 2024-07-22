from torch import nn
from torch.nn import functional as F
import torch
from .linear import Linear
import math


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, args):
        super(MultiheadAttention, self).__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query_linear = Linear(nn.Linear(embed_size, embed_size), args)
        self.key_linear = Linear(nn.Linear(embed_size, embed_size), args)
        self.value_linear = Linear(nn.Linear(embed_size, embed_size), args)

        self.fc_out = Linear(nn.Linear(embed_size, embed_size), args)

    def forward(self, q, k, v, attn_mask=None):
        N = q.shape[1]  # Batch size
        query_len, key_len, value_len = q.shape[0], k.shape[0], v.shape[0]

        query = self.query_linear(q)
        key = self.key_linear(k)
        value = self.value_linear(v)

        query = query.view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        #     seq_len, batch_size = query.shape[:2]

        #     # Always expands attn_mask to 4D
        #     if attn_mask.dim() == 3:
        #         attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
        #     else:  # attn_mask.dim() == 2:
        #         attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
        #             batch_size, self.num_heads, -1, -1
        #         )
        #     attn_mask = attn_mask_expanded

        out, attn_weights = scaled_dot_product_attention(query, key, value, attn_mask)
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)
        out = self.fc_out(out)
        out = out.transpose(0, 1)

        return out, attn_weights
