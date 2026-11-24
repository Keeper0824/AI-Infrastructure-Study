# 文档作者 brief B：M03 Paged KV + M07 Prefix Caching

## 通用要求

工作区：`/home/dictator/ai_infra/nanovllm`

先读：

1. `docs/00_architecture.md`（§5–§8，尤其 §6.4 KV 契约）
2. `docs/00_facts.md`（行号锚点、Qwen3 KV 尺寸、写作约定）
3. `docs/M00_contracts.md`（风格样例）
4. `docs/_template.md`
5. `ROADMAP.md`

只写：

- `docs/M03_paged_kv.md`
- `docs/M07_prefix_caching.md`

不要改代码/其他文档。每个文件 350–500 行。中文。

引用格式：`vllm/v1/core/block_pool.py:136 · class BlockPool`。新增引用先 `grep -n` 验证。
不要建议通读 `kv_cache_manager.py`（922 行）或 `block_pool.py`（901 行）；只读主路径。

---

## M03：Paged KV Cache：BlockPool + KVCacheManager + BlockTable

### 目标

把 M02 的连续 KV cache 换成**分页 KV cache**：

- `BlockPool` 管理固定大小的物理块；
- `KVCacheManager` 按请求持有块、分配/释放、维护引用计数；
- `BlockTable` 把逻辑块号映射到物理块号；
- `slot_mapping` 把本步每个 token 映射到 `block_id * block_size + offset`；
- `reshape_and_cache` 把 k/v 写进分页 tensor。

M03 **先不做前缀缓存**（`enable_caching=False`）；M07 再打开。这样可以把“分页”和
“前缀复用”两个复杂度分开。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/core/kv_cache_utils.py:176` | `KVCacheBlock` | `block_id` / `ref_cnt` / `block_hash` / `prev_free` / `next_free` |
| `vllm/v1/core/kv_cache_utils.py:241` | `KVCacheBlockCopy` | 可以跳过，M07 的 COW 才用 |
| `vllm/v1/core/kv_cache_utils.py:252` 附近 | `FreeKVCacheBlockQueue` | 侵入式双向链表：O(1) remove、LRU 顺序 |
| `vllm/v1/core/block_pool.py:136` | `BlockPool` | `blocks` / `free_block_queue` / `cached_block_hash_to_block` |
| `vllm/v1/core/block_pool.py:666` | `get_new_blocks()` | 从 free queue 取块、`ref_cnt += 1`、清理旧 hash |
| `vllm/v1/core/block_pool.py:752` | `touch()` | 命中缓存时 `ref_cnt += 1`，必要时从 free queue 移除 |
| `vllm/v1/core/block_pool.py:774` | `free_blocks()` | `ref_cnt -= 1`；归零后按 LRU 放回 |
| `vllm/v1/core/kv_cache_manager.py:131` | `KVCacheManager` | `req_to_blocks`、`block_pool`、`enable_caching` |
| `vllm/v1/core/kv_cache_manager.py:264` | `get_computed_blocks()` | M03 只看“没有缓存时返回空”；M07 细读 |
| `vllm/v1/core/kv_cache_manager.py:371` | `allocate_slots()` | 先看签名和 Blocks layout 注释；主路径见下 |
| `vllm/v1/core/kv_cache_manager.py:620` | `free()` | 倒序释放，尾块先被驱逐 |
| `vllm/v1/core/kv_cache_manager.py:782` | `cache_blocks()` | M07 用；M03 只看 `enable_caching=False` 的分支 |
| `vllm/v1/worker/block_table.py:49` | `BlockTable` | CPU/GPU 两份 block table，`append_row` / `add_row` |
| `vllm/v1/worker/block_table.py:153` | `compute_slot_mapping()` | `slot = block_id * block_size + pos % block_size` |
| `vllm/v1/worker/gpu_worker.py:634` | `determine_available_memory()` | 显存 profiling 的简化版 |
| `vllm/v1/worker/gpu_model_runner.py:7055` | `initialize_kv_cache()` | 每层 KV tensor 的分配；只看骨架 |
| `vllm/v1/kv_cache_interface.py:578` | `FullAttentionSpec` | `page_size_bytes` 公式 |
| `vllm/v1/kv_cache_interface.py:1460` | `KVCacheConfig` | 多 group；nano 只要单 group |

### 必须讲清的概念

1. **为什么分页**：连续 cache 按 `max_model_len` 预留，浪费；分页按需分配，块可复用。
   用 Qwen3-0.6B 算一笔账：每 token 每层 `2 * 8 * 128 * 2B = 4KB`；28 层 = 112KB/token；
   4K context 就是 ~448MB/请求。16GB 卡上 10 个并发就 4.5GB。
2. **block / block table / slot mapping** 三个概念的区别：
   - block 是物理页；
   - block table 是“逻辑块号 → 物理块号”；
   - slot 是“本步 token → 物理地址”，`block_id * block_size + offset`。
3. **ref_cnt**：一个块可以被多个请求共享（M07 前缀缓存）。`ref_cnt == 0` 才能回到 free queue。
4. **free queue 的 LRU 顺序**：队首是最久未用、最先被驱逐；`free_blocks` 把“有 hash 的块”
   放队尾（保留更久），“没 hash 的块”放队首（快速复用）。M03 可以先简化成统一放队尾，
   但文档要说明 vLLM 的差异。
5. **allocation failure**：`allocate_slots()` 返回 `None` 不是异常；Scheduler 据此抢占。
6. **内部碎片**：最后一个块可能只填一半；`num_computed_tokens // block_size` 是满块数。
7. **显存 sizing**：`num_gpu_blocks = available_kv_bytes // page_size_bytes`；
   `page_size_bytes = block_size * num_kv_heads * 2 * head_dim * dtype_size`。

### 必须实现的接口

严格按 `docs/00_architecture.md` §6.4：

```python
# nanovllm/core/kv_cache_utils.py
@dataclass(eq=False)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: bytes | None = None
    prev_free: "KVCacheBlock | None" = None
    next_free: "KVCacheBlock | None" = None

class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]): ...
    def popleft_n(self, n: int) -> list[KVCacheBlock]: ...
    def append_n(self, blocks: list[KVCacheBlock]) -> None: ...
    def prepend_n(self, blocks: list[KVCacheBlock]) -> None: ...
    def remove(self, block: KVCacheBlock) -> None: ...
    def __len__(self) -> int: ...

# nanovllm/core/block_pool.py
class BlockPool:
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]: ...
    def free_blocks(self, blocks: list[KVCacheBlock]) -> None: ...
    def touch(self, blocks: list[KVCacheBlock]) -> None: ...
    def get_cached_block(self, block_hash: bytes) -> KVCacheBlock | None: ...
    def cache_full_blocks(self, request, blocks, num_cached, num_full, block_size): ...
    @property
    def num_free_blocks(self) -> int: ...

# nanovllm/core/kv_cache_manager.py
class KVCacheManager:
    def get_computed_blocks(self, request) -> tuple[KVCacheBlocks, int]: ...
    def allocate_slots(self, request, num_new_tokens,
                       num_new_computed_tokens=0,
                       new_computed_blocks=None) -> KVCacheBlocks | None: ...
    def free(self, request) -> None: ...
    def cache_blocks(self, request, num_computed_tokens) -> None: ...
    def get_block_ids(self, request_id) -> list[int]: ...

# nanovllm/worker/block_table.py
class BlockTable:
    def add_row(self, block_ids: list[int], row_idx: int) -> None: ...
    def append_row(self, block_ids: list[int], row_idx: int) -> None: ...
    def clear_row(self, row_idx: int) -> None: ...
    def move_row(self, src: int, tgt: int) -> None: ...
    def compute_slot_mapping(self, query_start_loc, positions) -> torch.Tensor: ...
```

`KVCacheBlocks` 只需要 `blocks: list[KVCacheBlock]` + `get_block_ids()`。

### `allocate_slots()` 的简化算法（文档里要一步步写）

```text
输入：request, num_new_tokens, num_new_computed_tokens, new_computed_blocks
1. 如果 new_computed_blocks 非空：
     - 把它们的 ref_cnt += 1（touch）
     - 追加到 req_to_blocks[req_id]
2. total_computed = request.num_computed_tokens + num_new_computed_tokens
3. num_tokens_need = total_computed + num_new_tokens
4. num_blocks_need = ceil(num_tokens_need / block_size)
5. num_blocks_have = len(req_to_blocks[req_id])
6. num_new_blocks = num_blocks_need - num_blocks_have
7. 如果 num_new_blocks > block_pool.num_free_blocks: return None
8. blocks = block_pool.get_new_blocks(num_new_blocks)
9. req_to_blocks[req_id].extend(blocks)
10. return KVCacheBlocks(blocks)   # 只返回本步新块
```

注意：`request.num_computed_tokens` 是**本步调度前**的值；Scheduler 在 schedule 之后才
`+= num_scheduled_tokens`。文档必须强调这个顺序，否则会多算/少算块。

### 测试与对拍

- `test_m03_block_pool.py`：
  - 分配 N 个块，`ref_cnt==1`，free 后回队列；
  - 重复分配/释放 100 轮，块数守恒；
  - `num_free_blocks` 正确；
  - 分配超过剩余块返回错误（或 `None`，由 manager 处理）；
  - free queue 的 LRU 顺序（队首先出）。
- `test_m03_slot_mapping.py`：
  - 给定 block_table、positions、query_start_loc，手算 slot；
  - 边界：block 边界（pos=15/16）、多请求、prompt 长度不是 block_size 整数倍。
- `test_m03_gpu_kv.py`（`@pytest.mark.gpu`）：
  - 分配 2 个块，写入已知 k/v，按 slot 读回，`allclose`；
  - 不同 block 的同一 offset 不串。
- 端到端：M02 的 prompt 用 paged cache 跑 prefill + 1 token decode，logits 和连续 cache 一致。

### 常见坑

- `num_blocks_need` 用 `ceil(total_tokens / block_size)`，不是 `total_tokens // block_size`；
- `num_computed_tokens` 是调度前还是调度后，搞反；
- 释放时没有倒序，导致 LRU 顺序不对；
- `ref_cnt` 减到 0 后没有清 `block_hash`（M07 会用到）；
- `block_table` 的 padding 行用了 0，而 0 是真实块；nano 不保留 null block，所以
  attention 只能读 `seq_len` 以内的位置；
- slot mapping 用了 `positions // block_size` 当物理块号，忘了查 block table；
- KV tensor 的 layout 和 kernel 的 stride 假设不一致（文档里定死一种 layout）。

### 自测题

1. block table 和 slot mapping 的区别？各解决什么问题？
2. 为什么 `allocate_slots` 失败要返回 `None` 而不是抛异常？
3. `ref_cnt` 在什么时机 +1 / -1？
4. 为什么释放请求时要倒序释放块？
5. Qwen3-0.6B 的 `page_size_bytes` 是多少？16GB 卡大约能放多少块？

### 教练流程

概念 20min → 读 BlockPool/BlockTable 40min → 写 free queue/block pool 50min →
写 manager 50min → 写 block table/slot mapping 30min → GPU 读写测试 30min → 复盘 15min。
先让用户写 `FreeKVCacheBlockQueue` 的单测，再写 manager；不要一上来写完整 900 行。

---

## M07：Prefix Caching

### 目标

打开 `enable_prefix_caching`，让相同前缀的请求复用已算好的 KV block：

- `Request.block_hashes` 链式哈希；
- `BlockPool.cached_block_hash_to_block` 查找；
- `get_computed_blocks()` 找最长前缀命中；
- `allocate_slots()` 追加/`touch` 命中块；
- 一个请求结束后，块不立即释放（`ref_cnt==0` 但仍在 cache），下次可命中；
- 分配新块时，如果块有旧 hash，先驱逐（LRU）；
- 统计命中 token 数 / 命中率。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/request.py:285` | `update_block_hashes()` | 增量哈希 |
| `vllm/v1/core/kv_cache_utils.py:118` | `NONE_HASH` | 根哈希 |
| `vllm/v1/core/kv_cache_utils.py:695` | `hash_block_tokens()` | `(parent, tokens, extra)` |
| `vllm/v1/core/kv_cache_utils.py:941` | `get_request_block_hasher()` | 只返回新满块 |
| `vllm/v1/core/block_pool.py:198` | `get_cached_block()` | hash → block |
| `vllm/v1/core/block_pool.py:226` | `cache_full_blocks()` | 把满块插入 hash map、设置 `block_hash` |
| `vllm/v1/core/block_pool.py:752` | `touch()` | 命中时 ref_cnt++、移出 free queue |
| `vllm/v1/core/block_pool.py:774` | `free_blocks()` | 归零后按“有无 hash”分两段放回 |
| `vllm/v1/core/block_pool.py:729` | `_maybe_evict_cached_block()` | 复用时清理旧 hash |
| `vllm/v1/core/kv_cache_manager.py:264` | `get_computed_blocks()` | 最长命中、`num_tokens-1`、touch |
| `vllm/v1/core/kv_cache_manager.py:371` | `allocate_slots()` | 命中块追加 + 新块分配 |
| `vllm/v1/core/kv_cache_manager.py:782` | `cache_blocks()` | 把满块写入 cache |
| `vllm/v1/core/sched/scheduler.py:930-1080` | waiting 前缀查询 | 只读 `get_computed_blocks` 的调用和 `num_computed_tokens` 设置 |
| `vllm/v1/core/sched/scheduler.py:1616` | `_update_after_schedule()` | 调度后推进 `num_computed_tokens` |

### 必须讲清的概念

1. **链式哈希**：块 i 的 hash = H(block[i-1].hash, tokens[i*B:(i+1)*B])。前缀相同才同 hash。
2. **只哈希满块**：尾块会变；`num_full_blocks = num_tokens // block_size`。
3. **命中长度上限 `num_tokens - 1`**：全命中时没有 logits，必须至少重算最后一个 token。
   vLLM 会向下取整到 block 边界，可能重算一整块。
4. **ref_cnt 与 cache 的区别**：`ref_cnt==0` 的块仍在 hash map 里，可以被命中；只有被
   分配去装新数据时才驱逐。
5. **LRU**：`free_blocks` 把有 hash 的块放队尾（慢点被复用），无 hash 的放队首。
6. **cache 写入时机**：vLLM 在 `allocate_slots()` 末尾就 `cache_blocks()`（此时 KV 还没算）；
   nano 可以在 step 成功后 cache，文档要说明这个简化，以及为什么不会错（同步执行）。
7. **partial tail / COW**：vLLM 0.32 支持 partial block / CoW；nano 只做 full block，
   明确说“不支持 sub-block 命中”。

### 实现任务

T1 打开开关，构造 block hasher；T2 `BlockPool.get_cached_block` + `cached` map；
T3 `get_computed_blocks`（含 `num_tokens-1`）；T4 `allocate_slots` 追加/touch；
T5 `cache_full_blocks`（step 成功后）；T6 驱逐与 LRU；T7 指标；T8 测试/benchmark。

### 测试与对拍

- 两个请求共享 8 token 前缀（block_size=4）：
  - 第二个请求 `get_computed_blocks` 返回 8；
  - 第二个请求 `num_scheduled_tokens` 只剩后缀；
  - 输出与不开缓存完全一致。
- 前缀分叉：`[A][B]` vs `[C][B]`，第二块不能共享。
- 尾块不命中：prompt 长度不是 block_size 整数倍。
- 全命中：`num_tokens-1` 限制生效，至少重算 1 个 token（可能 1 块）。
- 驱逐：KV 紧张时，旧块被复用，hash 被清理，不能命中脏数据。
- 引用计数：两个请求共享块，一个结束后另一个仍能读；两个都结束后块回 free queue。
- 性能：相同 system prompt 的 8 个请求，第二个起 TTFT 明显下降。

### 常见坑

- 哈希了尾块；
- `get_computed_blocks` 没有 `num_tokens - 1`，全命中时 logits 是旧的/空的；
- 命中块没有 `touch`，被别的请求驱逐；
- `cache_full_blocks` 在 KV 写入前调用，但同步执行没问题；如果以后异步，要移到 step 后；
- `free_blocks` 把有 hash 的块放错队首，LRU 失效；
- 驱逐时只从 free queue 移除，忘了从 `cached` map 删除，导致命中已复用的块；
- 多个请求共享块，`free` 时直接清 hash，另一个请求还在用。

### 自测题

1. 为什么块哈希要带 parent？举一个反例。
2. 为什么 `get_computed_blocks` 要把命中长度限制在 `num_tokens - 1`？
3. `ref_cnt == 0` 的块为什么还能被命中？
4. `free_blocks` 为什么把有 hash 的块放队尾？
5. 如果两个请求共享一个块，什么时候才能真正复用这块的显存？

### 教练流程

概念 20min → 读 hash/BlockPool 40min → 写 get_computed_blocks 40min → 写 cache/evict 50min →
测试/benchmark 40min → 复盘 15min。先做“两个请求共享前缀”的最小 case，再做驱逐。

---

## 两篇文档的衔接

- M03 结尾：分页 cache 能跑通，但 `enable_caching=False`；下一站 M04（paged attention），
  M07 再回来开缓存。
- M07 依赖 M03 的 `BlockPool` / `KVCacheManager` 接口；不要改签名。
- M07 的测试要复用 M03 的 block pool 测试工具；文档里给出复用方式。

## 交付前检查

- [ ] 两个文件 350–500 行；
- [ ] `allocate_slots` 的 10 步算法写清楚；
- [ ] 有 Qwen3-0.6B 的 page size / block 数计算；
- [ ] 有“共享前缀 / 分叉 / 尾块 / 全命中 / 驱逐”五类测试；
- [ ] 明确 nano 不支持 partial block / COW；
- [ ] 没有建议通读 900 行文件。
