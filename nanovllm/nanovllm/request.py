"""Request：一个请求在整个生命周期里的可变状态。

对照 vLLM：
    vllm/v1/request.py:60    class Request
    vllm/v1/request.py:272   append_output_token_ids()
    vllm/v1/request.py:285   update_block_hashes()
    vllm/v1/request.py:299   num_tokens_with_spec
    vllm/v1/request.py:370   class RequestStatus

一句话：**Request 是 V1 里唯一“会动”的状态**。调度器每一步只是在改它的
`num_computed_tokens` / `status` / `block_hashes`，然后把它翻译成 SchedulerOutput。
"""

from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.hashing import BlockHash


class RequestStatus(enum.IntEnum):
    """请求状态。顺序有意义：PREEMPTED 之后的都算“已完成”。

    对应 vLLM 的 RequestStatus，这里删掉了多模态/结构化输出/streaming 等状态。
    """

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: "SamplingParams",
        arrival_time: float | None = 0.0,
        block_size: int = 16,
        block_hasher: "Callable[[Request], list[BlockHash]] | None" = None,
        client_index: int = 0,
        priority: int = 0,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params
        self.arrival_time = time.time() if arrival_time is None else arrival_time
        self.block_size = block_size
        self.max_tokens = sampling_params.max_tokens

        # ---- 调度器维护的状态 ----
        self.status = RequestStatus.WAITING
        # 已经算进 KV cache 的 token 数。V1 的核心变量：它追赶 num_tokens_with_spec。
        self.num_computed_tokens = 0
        # 被抢占次数（抢占时 num_computed_tokens 归零、status -> PREEMPTED）
        self.num_preemptions = 0
        # 投机解码的草稿 token（mini 版恒为空，但字段保留，保证公式与 V1 同构）
        self.spec_token_ids: list[int] = []

        # ---- token ----
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(self.prompt_token_ids)

        # ---- 前缀缓存 ----
        # 只记录“满块”的链式哈希，长度 == 已哈希的块数
        self.block_hashes: list[BlockHash] = []
        self._block_hasher = block_hasher
        if block_hasher is not None:
            self.update_block_hashes()

    # ------------------------------------------------------------------ #
    # 只读视图
    # ------------------------------------------------------------------ #
    @property
    def all_token_ids(self) -> list[int]:
        """prompt + 已生成 token（不含 spec）。注意返回的是内部 list 本身。"""
        return self._all_token_ids

    @property
    def output_token_ids(self) -> list[int]:
        return self._output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        """调度器要“追平”的目标：prompt + output + 草稿 token。

        对应 vllm/v1/request.py:299。
        """
        # TODO(M00-3): 一行搞定
        return self.num_tokens + len(self.spec_token_ids)
        # raise NotImplementedError("M00-3: implement Request.num_tokens_with_spec")

    # ------------------------------------------------------------------ #
    # 变更
    # ------------------------------------------------------------------ #
    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        """追加新生成的 token。

        对应 vllm/v1/request.py:272。注意两件事：
          1. 要同时更新 ``_output_token_ids`` 和 ``_all_token_ids``；
          2. 追加完必须调用 ``update_block_hashes()``——新 token 可能凑满一个新块。
        """
        # TODO(M00-4): 支持 int 和 list 两种入参
        if isinstance(token_ids, int):
            self._all_token_ids.append(token_ids)
            self._output_token_ids.append(token_ids)
        elif isinstance(token_ids, list):
            self._all_token_ids.extend(token_ids)
            self._output_token_ids.extend(token_ids)
        else:
            raise TypeError("token_ids非法")

        self.update_block_hashes()

        # raise NotImplementedError("M00-4: implement Request.append_output_token_ids")

    def update_block_hashes(self) -> None:
        """把“新增的满块”哈希追加到 ``self.block_hashes``。

        对应 vllm/v1/request.py:285：增量逻辑全部委托给 ``_block_hasher``，
        这里只负责 extend（这也是为什么 hasher 只能返回新块）。
        """
        # TODO(M00-5): 没 hasher 时直接返回
        if self._block_hasher is None:
            return
        self.block_hashes.extend(self._block_hasher(self))
        # raise NotImplementedError("M00-5: implement Request.update_block_hashes")

    # ------------------------------------------------------------------ #
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(
        self,
    ) -> "FinishReason | None":  # pyright: ignore[reportUndefinedVariable]
        """把 RequestStatus 映射成前端使用的 FinishReason。"""
        from nanovllm.outputs import FinishReason

        return {
            RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
            RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
            RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
            RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
        }.get(self.status)

    def reset_for_preemption(self) -> None:
        """抢占时重置调度状态（物理块由 KVCacheManager 释放）。

        对应 vllm/v1/core/sched/scheduler.py:1571 · _preempt_request() 的核心几行。
        """
        self.status = RequestStatus.PREEMPTED
        self.num_computed_tokens = 0
        self.num_preemptions += 1
        self.spec_token_ids.clear()

    def __lt__(self, other: "Request") -> bool:
        """优先级队列用：priority 小的先；同 priority 按到达时间。"""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.request_id < other.request_id

    def __repr__(self) -> str:
        return (
            f"<Request {self.request_id} status={self.status.name} "
            f"tokens={self.num_tokens} computed={self.num_computed_tokens}>"
        )
