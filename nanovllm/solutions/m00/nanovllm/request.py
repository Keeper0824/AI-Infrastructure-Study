"""M00 参考解：Request。卡住 30 分钟以上再看。"""

from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.hashing import BlockHash


class RequestStatus(enum.IntEnum):
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
        arrival_time: float = 0.0,
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

        self.status = RequestStatus.WAITING
        self.num_computed_tokens = 0
        self.num_preemptions = 0
        self.spec_token_ids: list[int] = []

        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(self.prompt_token_ids)

        self.block_hashes: list[BlockHash] = []
        self._block_hasher = block_hasher
        if block_hasher is not None:
            self.update_block_hashes()

    @property
    def all_token_ids(self) -> list[int]:
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
        return self.num_tokens + len(self.spec_token_ids)

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        if self._block_hasher is None:
            return
        self.block_hashes.extend(self._block_hasher(self))

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> "FinishReason | None":
        """把 RequestStatus 映射成前端使用的 FinishReason。"""
        from nanovllm.outputs import FinishReason

        return {
            RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
            RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
            RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
            RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
        }.get(self.status)

    def reset_for_preemption(self) -> None:
        self.status = RequestStatus.PREEMPTED
        self.num_computed_tokens = 0
        self.num_preemptions += 1
        self.spec_token_ids.clear()

    def __lt__(self, other: "Request") -> bool:
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
