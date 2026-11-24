import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_scaling: dict | None = None

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Qwen3Config":
        with open(f"{model_dir}/config.json", encoding="utf-8") as f:
            raw = json.load(f)

        assert raw["hidden_act"] == "silu"
        assert raw["attention_bias"] is False
        assert raw.get("rope_scaling") is None

        head_dim = (
            raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]
        )

        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=head_dim,
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            rope_scaling=raw.get("rope_scaling"),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x = (
            (
                x_fp32
                * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            ).to(self.weight.dtype)
            * self.weight
        ).to(x.dtype)
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        i = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (i / head_dim))
        pos = torch.arange(0, max_position, 1, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[positions]  # type: ignore
        sin = self.sin_cached[positions]  # type: ignore

        cos = cos[:, None]
        sin = sin[:, None]

        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, rotary_emb: RotaryEmbedding | None = None):
        super().__init__()
        self.num_heads = config.num_attention_heads  # 16
        self.num_kv_heads = config.num_key_value_heads  # 8
        self.head_dim = config.head_dim  # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 2

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )  # [2048,1024]
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )  # [1024,1024]
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )  # [1024,2048]

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary_emb = rotary_emb or RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T, 1024]
        positions: torch.Tensor,  # [T]
    ) -> torch.Tensor:  # [T, 1024]
        seq_len = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        k = k.repeat_interleave(
            self.num_kv_groups, dim=1
        )  # [seq_len,num_head,head_size]
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        q = q.transpose(0, 1).unsqueeze(0)  # [1,num_head,seq_len,head_size]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(seq_len, -1)
        return self.o_proj(o)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, rotary_emb: RotaryEmbedding | None = None):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, rotary_emb)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, rotary_emb=self.rotary_emb)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return x


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def load_weights(self, model_dir: str) -> list[str]:
        from nanovllm.models.weight_loader import load_model_weights

        loaded = load_model_weights(self, model_dir)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            assert (
                self.lm_head.weight.data_ptr()
                == self.model.embed_tokens.weight.data_ptr()
            )

        return loaded

    def forward(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return self.compute_logits(hidden_states)
