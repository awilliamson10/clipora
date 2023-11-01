import torch

from clipora.lora.linear import LoraInjectedLinear
from clipora.utils import _find_modules


def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module={"MultiheadAttention"},
):
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))

    torch.save(weights, path)


def extract_lora_ups_down(model, target_replace_module):
    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear],
    ):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras
