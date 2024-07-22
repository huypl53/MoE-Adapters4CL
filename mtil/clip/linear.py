from torch import nn
from .lora import LoRA


class Linear(nn.Module):
    def __init__(self, base_layer: nn.Module, emb_n=512, args=None):
        super(Linear, self).__init__()
        self.base_layer_ = base_layer
        lora_n_emd = emb_n
        # lora_n_emd = args.lin_lora_w
        lora_rank = args.lin_lora_r
        lora_scale = args.lin_lora_scale
        self.lora = LoRA(
            lora_n_emd, lora_rank, dropout=0.2, scale=lora_scale, device=args.device
        )

    def forward(self, x):
        residual = self.base_layer_(x)
        lora_x = self.lora.forward(x, residual)
        return lora_x
