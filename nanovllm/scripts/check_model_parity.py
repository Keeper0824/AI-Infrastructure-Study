import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from nanovllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from transformers import AutoTokenizer
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Check model parity")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径，例如 models/Qwen3-0.6B",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="输入 prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="最多生成多少 token",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help="绝对误差容忍度",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="相对误差容忍度",
    )

    return parser.parse_args()


def nano_greedy(nano, ids, max_new_tokens, eos, device):
    ids = ids.tolist()
    out = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            cur = torch.tensor(ids, device=device)
            logits = nano(cur, torch.arange(len(ids), device=device))
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            ids.append(nxt)
            if eos is not None and nxt == eos:
                break

    return out


def capture(modules, run):
    """Run `run()` once and capture the outputs of `modules`."""
    captured = {}
    handles = []

    for name, module in modules.items():

        def hook(_module, _inputs, output, name=name):
            if isinstance(output, tuple):
                output = output[0]
            captured[name] = output.detach()

        handles.append(module.register_forward_hook(hook))

    try:
        result = run()
    finally:
        for handle in handles:
            handle.remove()

    return result, captured


def check_stage(name, hf_value, nano_value, atol, rtol):
    hf_value = hf_value.float()
    nano_value = nano_value.float()

    if hf_value.shape != nano_value.shape:
        print(
            f"[fail] {name}_shape = "
            f"{tuple(hf_value.shape)} vs {tuple(nano_value.shape)}"
        )
        return False

    diff = (hf_value - nano_value).abs().max().item()

    try:
        torch.testing.assert_close(hf_value, nano_value, atol=atol, rtol=rtol)
    except AssertionError:
        print(f"[fail] {name}_max_abs_diff = {diff:.5f}")
        return False

    print(f"{name}_max_abs_diff = {diff:.5f}")
    return True


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("model =", args.model)
    print("prompt =", args.prompt)
    print("max_new_tokens =", args.max_new_tokens)
    print("atol =", args.atol)
    print("rtol =", args.rtol)

    tok = AutoTokenizer.from_pretrained(args.model)

    hf = HFQwen3ForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        # attn_implementation="eager",
    ).to(device)  # type: ignore
    hf.eval()

    cfg = Qwen3Config.from_pretrained(args.model)
    nano = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    nano.load_weights(args.model)
    nano.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids[0].to(device)
    pos = torch.arange(ids.numel(), device=device)
    attn_mask = torch.ones_like(ids[None])

    hf_modules = {
        "embedding": hf.model.embed_tokens,
        "layer0": hf.model.layers[0],
        "final_norm": hf.model.norm,
    }
    nano_modules = {
        "embedding": nano.model.embed_tokens,
        "layer0": nano.model.layers[0],
        "final_norm": nano.model.norm,
    }

    attn_mask = torch.ones_like(ids[None])

    with torch.no_grad():
        hf_logits, hf_cap = capture(
            hf_modules, lambda: hf(ids[None], attention_mask=attn_mask).logits
        )
        nano_logits, nano_cap = capture(nano_modules, lambda: nano(ids, pos))

    stages = [
        ("embedding", hf_cap["embedding"][0], nano_cap["embedding"]),
        ("layer0", hf_cap["layer0"][0], nano_cap["layer0"]),
        ("final_norm", hf_cap["final_norm"][0], nano_cap["final_norm"]),
        ("logits", hf_logits[0, -1], nano_logits[-1]),
    ]

    for name, hf_value, nano_value in stages:
        if not check_stage(name, hf_value, nano_value, args.atol, args.rtol):
            return 1

    with torch.no_grad():
        hf_out = hf.generate(  # type: ignore
            ids[None],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            attention_mask=attn_mask,
        )[0, ids.numel() :].tolist()

        nano_out = nano_greedy(nano, ids, args.max_new_tokens, tok.eos_token_id, device)

    greedy_16_match = nano_out == hf_out
    if not greedy_16_match:
        print("greedy_16_match = False")
        return 1

    print("greedy_16_match = True")
    print("[ok] parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
