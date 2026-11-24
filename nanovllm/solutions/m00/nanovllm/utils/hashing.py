"""M00 参考解：块哈希。卡住 30 分钟以上再看。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanovllm.request import Request

BlockHash = bytes
NONE_HASH: BlockHash = hashlib.sha256(b"nanovllm-none-hash").digest()


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: list[int] | tuple[int, ...],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    parent = NONE_HASH if parent_block_hash is None else parent_block_hash
    digest = hash_function(
        (parent, tuple(curr_block_token_ids), extra_keys)
    )
    return bytes(digest)


def get_request_block_hasher(
    block_size: int,
    hash_function: Callable[[Any], bytes],
) -> Callable[["Request"], list[BlockHash]]:
    def hasher(request: "Request") -> list[BlockHash]:
        new_hashes: list[BlockHash] = []
        parent = request.block_hashes[-1] if request.block_hashes else None
        num_full_blocks = request.num_tokens // block_size
        for i in range(len(request.block_hashes), num_full_blocks):
            tokens = request.all_token_ids[i * block_size : (i + 1) * block_size]
            parent = hash_block_tokens(hash_function, parent, tokens)
            new_hashes.append(parent)
        return new_hashes

    return hasher
