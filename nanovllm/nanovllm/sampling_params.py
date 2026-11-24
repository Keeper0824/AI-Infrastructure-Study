"""采样参数：前端传给引擎的“怎么生成”配置。

对照 vLLM：`vllm/sampling_params.py`（类 `SamplingParams`）。
nano 版只保留本课程会用到的字段；`n` 先保留但不实现并行采样。

注意：`eos_token_id` 在 vLLM 里由 InputProcessor 从 tokenizer/config 填入，
不是用户参数。nano 版也由 LLM 初始化时填入。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    # 生成长度
    max_tokens: int = 16
    n: int = 1

    # 采样
    temperature: float = 0.0  # 0.0 = greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 = 不限制
    min_p: float = 0.0
    seed: int | None = None

    # 停止
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False

    # 惩罚
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # 输出
    logprobs: int | None = None  # None=不返回；0=只返回采样 token；-1=全部
    skip_special_tokens: bool = True

    # 由 tokenizer / model config 填入
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f"top_k must be -1 or >= 1, got {self.top_k}")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )
        if self.logprobs is not None and self.logprobs < -1:
            raise ValueError(f"logprobs must be >= -1, got {self.logprobs}")
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        elif self.stop is not None:
            self.stop = list(self.stop)
        if self.stop_token_ids is not None:
            self.stop_token_ids = list(self.stop_token_ids)

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def max_num_logprobs(self) -> int | None:
        return self.logprobs
