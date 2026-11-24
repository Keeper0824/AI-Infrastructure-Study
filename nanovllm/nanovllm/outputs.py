"""输出数据结构：从 ModelRunner 一路回传到前端。

对照 vLLM：
    vllm/v1/outputs.py:267              ModelRunnerOutput
    vllm/v1/engine/__init__.py:199      EngineCoreOutput
    vllm/v1/engine/__init__.py:259      EngineCoreOutputs
    vllm/v1/engine/__init__.py:48       FinishReason
    vllm/engine/request_output.py       CompletionOutput / RequestOutput
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class FinishReason(enum.IntEnum):
    """请求结束原因。值要和 OpenAI 的 finish_reason 字符串对得上。"""

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3

    def __str__(self) -> str:
        return ("stop", "length", "abort", "error")[self.value]


@dataclass
class LogprobsLists:
    """CPU 侧的 logprobs，按请求分组，方便直接返回给前端。"""

    token_ids: list[list[int]] = field(default_factory=list)
    logprobs: list[list[float]] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)


@dataclass
class ModelRunnerOutput:
    """ModelRunner 本步的产出。**不可变**，会被序列化回 Scheduler。"""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    logprobs: LogprobsLists | None = None


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})


@dataclass
class EngineCoreOutput:
    """Scheduler 回给前端的一个请求增量。"""

    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    new_logprobs: LogprobsLists | None = None


@dataclass
class EngineCoreOutputs:
    outputs: list[EngineCoreOutput] = field(default_factory=list)


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = False
