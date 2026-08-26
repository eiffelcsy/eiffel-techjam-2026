"""Exponential moving average of the adapter weights.

Cheap variance reduction on a small model trained for few steps, and it gives
two checkpoints per run -- raw and EMA -- for the weight soup at no extra
compute. Evaluate both; report whichever, but say which.
"""

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow, live in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if shadow.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(live.detach(), alpha=1 - self.decay)
            else:
                shadow.copy_(live)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow.state_dict())

    def state_dict(self) -> dict:
        return self.shadow.state_dict()
