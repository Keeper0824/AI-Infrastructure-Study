"""M00-3 / M00-4 / M00-5：Request 的计数、追加与哈希联动。"""

import hashlib

import pytest

from nanovllm.outputs import FinishReason
from nanovllm.request import Request, RequestStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.hashing import get_request_block_hasher


def sha256_of(obj) -> bytes:
    return hashlib.sha256(repr(obj).encode()).digest()


def make_request(prompt, block_size=4, with_hasher=True) -> Request:
    hasher = get_request_block_hasher(block_size, sha256_of) if with_hasher else None
    return Request(
        "r0",
        prompt,
        SamplingParams(max_tokens=8),
        block_size=block_size,
        block_hasher=hasher,
    )


# --------------------------------------------------------------------------- #
# 计数
# --------------------------------------------------------------------------- #
def test_initial_counts():
    r = make_request([1, 2, 3], with_hasher=False)
    assert r.num_tokens == 3
    assert r.num_output_tokens == 0
    assert r.num_tokens_with_spec == 3
    assert r.num_computed_tokens == 0
    assert r.status is RequestStatus.WAITING
    assert r.max_tokens == 8


def test_num_tokens_with_spec_counts_drafts():
    r = make_request([1, 2, 3], with_hasher=False)
    r.spec_token_ids = [98, 99]
    assert r.num_tokens == 3, "草稿 token 不算进 num_tokens"
    assert r.num_output_tokens == 0
    assert r.num_tokens_with_spec == 5, "num_tokens_with_spec = num_tokens + len(spec)"


# --------------------------------------------------------------------------- #
# 追加 token
# --------------------------------------------------------------------------- #
def test_append_single_token():
    r = make_request([1, 2, 3], with_hasher=False)
    r.append_output_token_ids(7)
    assert r.all_token_ids == [1, 2, 3, 7]
    assert r.output_token_ids == [7]
    assert r.num_tokens == 4
    assert r.num_output_tokens == 1


def test_append_token_list():
    r = make_request([1, 2], with_hasher=False)
    r.append_output_token_ids([7, 8])
    assert r.all_token_ids == [1, 2, 7, 8]
    assert r.output_token_ids == [7, 8]
    assert r.num_output_tokens == 2


def test_append_triggers_hasher():
    r = make_request([1, 2, 3], block_size=4)
    assert r.block_hashes == []
    r.append_output_token_ids(4)
    assert len(r.block_hashes) == 1, "追加第 4 个 token 后应凑满一个块"


def test_update_block_hashes_is_idempotent_without_new_tokens():
    r = make_request([1, 2, 3, 4, 5, 6, 7], block_size=4)
    snapshot = list(r.block_hashes)
    r.update_block_hashes()
    assert r.block_hashes == snapshot


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
def test_status_ordering_and_is_finished():
    assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)
    assert not RequestStatus.is_finished(RequestStatus.WAITING)
    assert not RequestStatus.is_finished(RequestStatus.RUNNING)
    assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)


def test_reset_for_preemption():
    r = make_request([1, 2, 3, 4])
    r.status = RequestStatus.RUNNING
    r.num_computed_tokens = 4
    r.spec_token_ids = [9]
    r.reset_for_preemption()
    assert r.status is RequestStatus.PREEMPTED
    assert r.num_computed_tokens == 0
    assert r.num_preemptions == 1
    assert r.spec_token_ids == []
    assert r.block_hashes, "抢占不应清空 token 的块哈希"


def test_output_token_ids_property():
    r = make_request([1, 2], with_hasher=False)
    r.append_output_token_ids(3)
    assert r.output_token_ids == [3]
    assert r.output_token_ids is r._output_token_ids


def test_sampling_params_validation():
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_k=0)
    with pytest.raises(ValueError):
        SamplingParams(logprobs=-2)


def test_finished_reason_mapping():
    r = make_request([1], with_hasher=False)
    assert r.get_finished_reason() is None
    r.status = RequestStatus.FINISHED_STOPPED
    assert r.get_finished_reason() is FinishReason.STOP
    r.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert r.get_finished_reason() is FinishReason.LENGTH
    r.status = RequestStatus.FINISHED_ABORTED
    assert r.get_finished_reason() is FinishReason.ABORT
