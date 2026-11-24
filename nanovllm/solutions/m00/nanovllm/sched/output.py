"""M00 参考解：SchedulerOutput。卡住 30 分钟以上再看。"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanovllm.sampling_params import SamplingParams


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: list[int]
    num_computed_tokens: int
    num_output_tokens: int = 0


@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str] = field(default_factory=set)
    new_block_ids: list[list[int] | None] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(req_ids=[])


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] = field(default_factory=set)

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        )

    def validate(self, max_num_batched_tokens: int) -> None:
        total = sum(self.num_scheduled_tokens.values())
        assert self.total_num_scheduled_tokens == total, (
            f"total_num_scheduled_tokens={self.total_num_scheduled_tokens} "
            f"!= sum(num_scheduled_tokens)={total}"
        )
        assert 0 <= total <= max_num_batched_tokens, (
            f"total_num_scheduled_tokens={total} exceeds "
            f"max_num_batched_tokens={max_num_batched_tokens}"
        )
        assert all(n > 0 for n in self.num_scheduled_tokens.values()), (
            "every scheduled request must have at least one token"
        )

        scheduled_ids = set(self.num_scheduled_tokens)
        assert not (self.finished_req_ids & scheduled_ids), (
            "finished requests must not be scheduled in the same step"
        )

        new_ids = [r.req_id for r in self.scheduled_new_reqs]
        assert len(new_ids) == len(set(new_ids)), "duplicate new request ids"
        assert set(new_ids) <= scheduled_ids, (
            f"new requests not scheduled: {set(new_ids) - scheduled_ids}"
        )

        cached = self.scheduled_cached_reqs
        assert len(cached.req_ids) == len(set(cached.req_ids)), (
            "duplicate cached request ids"
        )
        assert len(cached.req_ids) == len(cached.new_block_ids) == len(
            cached.num_computed_tokens
        ) == len(cached.num_output_tokens), (
            "CachedRequestData lists must be aligned"
        )
        assert set(cached.req_ids) <= scheduled_ids, (
            f"cached requests not scheduled: {set(cached.req_ids) - scheduled_ids}"
        )
        assert not (set(new_ids) & set(cached.req_ids)), (
            "a request cannot be both new and cached"
        )
