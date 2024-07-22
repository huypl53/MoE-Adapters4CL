from torch import nn, functional as F
import torch
from mtil.clip.linear import Linear


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

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

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]  # Batch size
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)

        out, attn_weights = scaled_dot_product_attention(query, key, value, mask)

        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)

        out = self.fc_out(out)

        return out, attn_weights
