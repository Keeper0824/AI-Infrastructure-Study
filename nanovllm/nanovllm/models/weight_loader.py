import json
import os

import torch
from safetensors import safe_open
from torch import nn


def load_model_weights(model: nn.Module, model_dir: str) -> list[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]

    state = {}
    for shard in shards:
        with safe_open(
            os.path.join(model_dir, shard), framework="pt", device="cpu"
        ) as f:
            for name in f.keys():
                state[name] = f.get_tensor(name)

    expected = {name for name, _ in model.named_parameters()}
    missing = expected - state.keys()
    unexpected = state.keys() - expected
    assert missing <= {"lm_head.weight"}, f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"

    loaded = []
    for name, p in model.named_parameters():
        if name not in state:
            continue
        t = state[name]
        assert p.shape == t.shape, (name, p.shape, t.shape)
        with torch.no_grad():
            p.data.copy_(t)
        loaded.append(name)

    return loaded


# from nanovllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM


# def main() -> int:
#     model_dir = "models/Qwen3-0.6B"
#     config = Qwen3Config.from_pretrained(model_dir)
#     model = Qwen3ForCausalLM(config)
#     load_model_weights(model, model_dir)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
