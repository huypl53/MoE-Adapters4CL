import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(
        self, n_emb: int, rank: int, dropout=0.1, scale="1.0", device: str = "cpu"
    ):
        super().__init__()
        self.n_emb = n_emb
        self.r = rank
        self._device = device

        self.lora_a_ = nn.Linear(
            self.n_emb,
            self.r_,
            bias=False,
            dtype=torch.float32,
            device=self.device_,
        )
        self.lora_b_ = nn.Linear(
            self.r_,
            self.n_emb,
            bias=False,
            dtype=torch.float32,
            device=self.device_,
        )

        if scale == "learnable_scalar":
            self.scale_ = nn.Parameter(torch.ones(1))
        else:
            self.scale_ = float(scale)

        self.dropout_ = self.dropout_ = nn.Dropout(dropout)

    def forward(self, x, residual=None):
        result_lora = (
            self.lora_b_(self.lora_a_(self.dropout_(x.to(torch.float32)))) * self.scale_
        )

        if residual:  # False
            return residual + result_lora.to(residual.dtype)
        else:
            return result_lora
