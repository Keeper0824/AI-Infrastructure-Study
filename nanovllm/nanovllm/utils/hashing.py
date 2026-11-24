"""块哈希：前缀缓存的“唯一键”。

对照 vLLM：
    vllm/v1/core/kv_cache_utils.py:118   NONE_HASH
    vllm/v1/core/kv_cache_utils.py:695   hash_block_tokens()
    vllm/v1/core/kv_cache_utils.py:941   get_request_block_hasher()

为什么是“链式”哈希
------------------
块 0 的哈希只由 token[0:B] 决定；
块 1 的哈希由 (块 0 的哈希, token[B:2B]) 决定；
以此类推。

于是“前缀相同”必然“哈希相同”，而“前缀里改了一个 token”会让**后面所有块**的哈希全变。
前缀缓存的正确性就靠这一点：只共享哈希完全一致的那些块。

为什么不能哈希尾块
------------------
尾块（partial block）还会长出新 token，哈希随长度变化。真 vLLM 只对
`num_tokens // block_size` 个满块做哈希，尾块留给下一步。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanovllm.request import Request

BlockHash = bytes

# 根哈希：第一个块的 parent 用这个常量，保证 parent 字段永远非空。
# 对应 vLLM 的 NONE_HASH（kv_cache_utils.py:118）。
NONE_HASH: BlockHash = hashlib.sha256(b"nanovllm-none-hash").digest()


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: list[int] | tuple[int, ...],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """把 (parent, 本块 token, extra) 压成一个 32 字节哈希。

    Args:
        hash_function: 任意 ``对象 -> bytes`` 的函数（真 vLLM 传 sha256）。
        parent_block_hash: 上一块的哈希；第一块传 ``None``。
        curr_block_token_ids: 本块的 token（调用方保证是满块）。
        extra_keys: 额外键（LoRA id / 多模态 hash 等），mini 版保留参数位。

    Returns:
        长度 32 的 bytes。
    """
    # TODO(M00-1):
    #   1) parent 为 None 时换成 NONE_HASH
    #   2) 把 (parent, tuple(tokens), extra_keys) 交给 hash_function
    #   3) 保证返回值是 bytes

    parent = NONE_HASH if parent_block_hash is None else parent_block_hash
    block_hash = hash_function((parent, tuple(curr_block_token_ids), extra_keys))
    return bytes(block_hash)

    # raise NotImplementedError("M00-1: implement hash_block_tokens")


def get_request_block_hasher(
    block_size: int,
    hash_function: Callable[[Any], bytes],
) -> Callable[["Request"], list[BlockHash]]:
    """构造一个**增量** hasher：只返回“新增满块”的哈希。

    返回值会被 ``Request.update_block_hashes()`` 用 ``extend`` 追加到
    ``request.block_hashes``，所以这里**只能返回新块**。

    必须满足：
      * 只对满块计算：``num_full_blocks = request.num_tokens // block_size``；
      * 从 ``len(request.block_hashes)`` 开始算，已算过的不重复；
      * 链式：第 i 块的 parent 是第 i-1 块的哈希
        （第一块如果前面已有哈希，parent 是 ``request.block_hashes[-1]``）。
    """

    def hasher(request: "Request") -> list[BlockHash]:
        # TODO(M00-2):
        #   new_hashes = []
        #   parent = request.block_hashes[-1] if request.block_hashes else None
        #   for i in range(len(request.block_hashes), request.num_tokens // block_size):
        #       tokens = request.all_token_ids[i * block_size:(i + 1) * block_size]
        #       parent = hash_block_tokens(hash_function, parent, tokens)
        #       new_hashes.append(parent)
        #   return new_hashes

        len_total = len(request.all_token_ids)
        hash_table_temp_len = len(request.block_hashes)
        startidx = hash_table_temp_len * block_size
        add_hash_table = []
        hash_before: BlockHash = (
            NONE_HASH if hash_table_temp_len == 0 else request.block_hashes[-1]
        )
        for i in range(startidx, len_total - block_size + 1, block_size):
            add_hash_table.append(
                hash_block_tokens(
                    hash_function=hash_function,
                    parent_block_hash=hash_before,
                    curr_block_token_ids=request.all_token_ids[i : i + block_size],
                )
            )
            hash_before = add_hash_table[-1]
        return add_hash_table
        # raise NotImplementedError("M00-2: implement incremental block hasher")

    return hasher
