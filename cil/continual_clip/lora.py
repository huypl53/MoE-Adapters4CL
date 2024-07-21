
import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Module, n_emb: int, rank: int, device: str):
        super().__init__()
        self.base_layer = base_layer
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

    def forward(self):
        pass
