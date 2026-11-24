"""M00-1 / M00-2：块哈希与增量 hasher 的验收测试。"""

import hashlib

from nanovllm.request import Request
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.hashing import (
    NONE_HASH,
    get_request_block_hasher,
    hash_block_tokens,
)


def sha256_of(obj) -> bytes:
    """测试用的 hash_function：与 vLLM 一样，接受任意对象返回 bytes。"""
    return hashlib.sha256(repr(obj).encode()).digest()


def make_request(prompt, block_size=4, request_id="r0") -> Request:
    return Request(
        request_id,
        prompt,
        SamplingParams(max_tokens=4),
        block_size=block_size,
        block_hasher=get_request_block_hasher(block_size, sha256_of),
    )


# --------------------------------------------------------------------------- #
# hash_block_tokens
# --------------------------------------------------------------------------- #
def test_hash_is_deterministic_and_32_bytes():
    h1 = hash_block_tokens(sha256_of, None, [1, 2, 3, 4])
    h2 = hash_block_tokens(sha256_of, None, [1, 2, 3, 4])
    assert h1 == h2
    assert isinstance(h1, bytes)
    assert len(h1) == 32


def test_parent_changes_hash():
    root = hash_block_tokens(sha256_of, None, [1, 2, 3, 4])
    child = hash_block_tokens(sha256_of, root, [1, 2, 3, 4])
    orphan = hash_block_tokens(sha256_of, b"\x01" * 32, [1, 2, 3, 4])
    assert root != child, "同样 token、不同 parent，哈希必须不同（链式）"
    assert child != orphan


def test_tokens_change_hash():
    a = hash_block_tokens(sha256_of, None, [1, 2, 3, 4])
    b = hash_block_tokens(sha256_of, None, [1, 2, 3, 5])
    assert a != b


def test_extra_keys_change_hash():
    a = hash_block_tokens(sha256_of, None, [1, 2, 3, 4])
    b = hash_block_tokens(
        sha256_of, None, [1, 2, 3, 4], extra_keys=("lora", 7)
    )
    assert a != b, "extra_keys 必须参与哈希（vLLM 用它区分 LoRA / 多模态）"


def test_none_parent_uses_none_hash():
    assert hash_block_tokens(sha256_of, None, [1, 2, 3, 4]) == hash_block_tokens(
        sha256_of, NONE_HASH, [1, 2, 3, 4]
    )


# --------------------------------------------------------------------------- #
# get_request_block_hasher
# --------------------------------------------------------------------------- #
def test_hasher_only_hashes_full_blocks():
    # 5 个 token、block_size=4 -> 只有 1 个满块
    r = make_request([1, 2, 3, 4, 5])
    assert len(r.block_hashes) == 1


def test_hasher_is_incremental_and_idempotent():
    r = make_request([1, 2, 3, 4, 5, 6, 7])  # 7 token -> 1 个满块
    assert len(r.block_hashes) == 1
    snapshot = list(r.block_hashes)

    r.update_block_hashes()  # 再调一次，不应该重复追加
    assert r.block_hashes == snapshot

    r.append_output_token_ids(8)  # 8 token -> 2 个满块
    assert len(r.block_hashes) == 2


def test_hasher_chains_parent():
    r = make_request([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(r.block_hashes) == 2
    expected_second = hash_block_tokens(
        sha256_of, r.block_hashes[0], [5, 6, 7, 8]
    )
    assert r.block_hashes[1] == expected_second, "第 2 块的 parent 必须是第 1 块的哈希"


def test_same_prefix_gets_same_hashes():
    a = make_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="a")
    b = make_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="b")
    assert a.block_hashes == b.block_hashes


def test_different_prefix_breaks_chain():
    a = make_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="a")
    b = make_request([1, 2, 3, 9, 5, 6, 7, 8], request_id="b")
    assert a.block_hashes[0] != b.block_hashes[0]
    assert a.block_hashes[1] != b.block_hashes[1], "前缀变了，后续块的哈希必须全变"


def test_partial_block_not_hashed_then_hashed_when_full():
    r = make_request([1, 2, 3], block_size=4)  # 3 token，0 个满块
    assert r.block_hashes == []
    r.append_output_token_ids(4)  # 4 token -> 1 个满块
    assert len(r.block_hashes) == 1
