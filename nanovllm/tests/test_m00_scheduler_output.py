"""M00-6：SchedulerOutput 的七条不变量。"""

import pytest

from nanovllm.sampling_params import SamplingParams
from nanovllm.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)


def new_req(req_id: str, n_blocks: int = 1) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(),
        block_ids=list(range(n_blocks)),
        num_computed_tokens=0,
    )


def make_output(
    num_scheduled_tokens=None,
    total_num_scheduled_tokens=None,
    **kwargs,
) -> SchedulerOutput:
    tokens = {"a": 4} if num_scheduled_tokens is None else num_scheduled_tokens
    total = (
        sum(tokens.values())
        if total_num_scheduled_tokens is None
        else total_num_scheduled_tokens
    )
    return SchedulerOutput(
        scheduled_new_reqs=kwargs.pop("scheduled_new_reqs", []),
        scheduled_cached_reqs=kwargs.pop(
            "scheduled_cached_reqs", CachedRequestData.make_empty()
        ),
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=total,
        **kwargs,
    )


def test_make_empty_is_consistent():
    out = SchedulerOutput.make_empty()
    assert out.total_num_scheduled_tokens == 0
    assert out.num_scheduled_tokens == {}
    out.validate(max_num_batched_tokens=8)


def test_valid_output_passes():
    make_output().validate(max_num_batched_tokens=8)


def test_total_mismatch_raises():
    out = make_output(total_num_scheduled_tokens=99)
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=128)


def test_over_budget_raises():
    out = make_output(num_scheduled_tokens={"a": 4, "b": 4})
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=4)


def test_zero_tokens_raises():
    out = make_output(num_scheduled_tokens={"a": 0, "b": 4})
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=8)


def test_finished_request_must_not_be_scheduled():
    out = make_output(finished_req_ids={"a"})
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=8)


def test_new_request_must_be_in_num_scheduled_tokens():
    out = make_output(scheduled_new_reqs=[new_req("b")])
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=8)


def test_new_request_included_is_valid():
    out = make_output(scheduled_new_reqs=[new_req("a")])
    out.validate(max_num_batched_tokens=8)


def test_duplicate_new_and_cached_raises():
    cached = CachedRequestData(
        req_ids=["a"],
        resumed_req_ids=set(),
        new_block_ids=[None],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )
    out = make_output(
        scheduled_new_reqs=[new_req("a")],
        scheduled_cached_reqs=cached,
    )
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=8)


def test_cached_fields_must_be_aligned():
    cached = CachedRequestData(
        req_ids=["a"],
        resumed_req_ids=set(),
        new_block_ids=[None],
        num_computed_tokens=[],  # 少一个
        num_output_tokens=[0],
    )
    out = make_output(scheduled_cached_reqs=cached)
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=8)


def test_cached_request_must_be_scheduled():
    cached = CachedRequestData(
        req_ids=["b"],
        resumed_req_ids=set(),
        new_block_ids=[None],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )
    out = make_output(scheduled_cached_reqs=cached)
    with pytest.raises(AssertionError):
        out.validate(max_num_batched_tokens=8)


def test_preempted_does_not_conflict_with_scheduling():
    # 被抢占的请求可以（也应该）在本步重新出现，二者不冲突
    out = make_output(preempted_req_ids={"b"})
    out.validate(max_num_batched_tokens=8)
