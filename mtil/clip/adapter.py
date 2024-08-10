# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math

import torch
import torch.nn as nn

from .fasterkan import FasterKAN as KAN


class Adapter(nn.Module):
    def __init__(
        self,
        d_model=None,
        bottleneck=None,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="1.0",
        adapter_layernorm_option="in",
        text_or_image="image",
    ):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.dropout = dropout
        self.text_or_image = text_or_image
        if self.text_or_image == "image":
            dim = d_model  # 768
            # hdim_kan = 192 // 2
            hdim_kan = 16
            # self.kan = KAN([dim, hdim_kan, dim])
            self.down_proj = KAN([dim, hdim_kan, dim])
            self.non_linear_func = nn.ReLU()
            self.up_proj = KAN([dim, hdim_kan, dim])
        else:
            self.down_proj = nn.Linear(self.n_embd, 64)
            self.non_linear_func = nn.ReLU()
            self.up_proj = nn.Linear(self.down_size, self.n_embd)

            if init_option == "bert":
                raise NotImplementedError
            elif init_option == "lora":
                with torch.no_grad():
                    nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                    nn.init.zeros_(self.up_proj.weight)
                    nn.init.zeros_(self.down_proj.bias)
                    nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":  #  none
            x = self.adapter_layer_norm_before(x)

        if self.text_or_image == "image":
            if x.shape[0] > 0:
                down = self.down_proj(x.reshape(-1, x.shape[-1])).reshape_as(x)
                down = self.non_linear_func(down)
                down = nn.functional.dropout(
                    down, p=self.dropout, training=self.training
                )
                up = self.up_proj(down.reshape(-1, down.shape[-1])).reshape_as(x)
            else:
                up = x

        else:
            down = self.down_proj(x)
            down = self.non_linear_func(down)
            down = nn.functional.dropout(down, p=self.dropout, training=self.training)
            up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == "out":  #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:  # False
            output = up + residual
        else:
            output = up
        return output
