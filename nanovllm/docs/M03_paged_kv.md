# M03 · Paged KV Cache：BlockPool / KVCacheManager / BlockTable

> **模块目标**：把 M02 的连续 KV cache 换成**分页 KV cache**——块由 `BlockPool` 管，
> 请求到块由 `KVCacheManager` 管，逻辑块到物理块由 `BlockTable` 管，本步 token 的
> 落点由 `slot_mapping` 管。
>
> **预计工时**：4–5 小时　**依赖**：M00（`Request`/块哈希）、M02（连续 cache 作 oracle）
> **测试**：CPU（块池/分配/slot）+ GPU（KV 读写/对拍）
>
> **产出**：`test_m03_block_pool.py`、`test_m03_slot_mapping.py`、`test_m03_gpu_kv.py`
> 全绿；同一 prompt 在连续 cache 与 paged cache 下 logits 一致。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m03_block_pool.py tests/test_m03_slot_mapping.py -q   # 20 passed
python -m pytest tests/test_m03_gpu_kv.py tests/test_m03_parity.py -q -m gpu      # 4 passed
```

五个能脱口而出的答案：连续 cache 在 4K 下每请求浪费多少显存？block / block table /
slot mapping 三者区别？`allocate_slots` 用调度前还是调度后的值？`ref_cnt==0` 的块为什么
不一定能立刻装新数据？Qwen3-0.6B 的 `page_size_bytes` 是多少？

**本模块不开前缀缓存**（`enable_caching=False`）：先把“分页 + 地址映射”做对，M07 再开缓存。

---

## 1. 为什么需要这个模块（10 分钟）

M02 的连续 cache 按 `max_model_len` 一次预留。Qwen3-0.6B（bf16，8 个 KV head，
head_dim 128，28 层）的账：

| 量 | 公式 | 数值 |
|---|---|---|
| 每 token 每层 K+V | `2 * 8 * 128 * 2B` | 4 KiB |
| 每 token 全模型 | `28 * 4 KiB` | 112 KiB |
| 4K 请求 | `4096 * 112 KiB` | 448 MiB |
| 10 个并发 4K 请求 | `10 * 448 MiB` | ~4.4 GiB |

而 4K 里平均可能只用 512 token（12.5%）：连续 cache 把“最大长度”当“实际长度”付费。
分页把 KV 切成固定大小 block（`block_size=16`），用到哪块给哪块，请求结束块回池子。
代价是多一层映射：attention 不能再假设“第 i 个 token 在 cache[i]”。对 M02 只变一处：

```text
M02:  attn(q, k, v, k_cache[req], v_cache[req])
M03:  attn(q, k, v, k_cache_pool, v_cache_pool, block_table, seq_lens)
```

M04 才改 attention；M03 先保证“能分配、能写对、能按 slot 读回”。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`
> 引用格式：`vllm/v1/...py:行号 · 符号`；行号漂了用 `grep -n "<符号>"` 重定位。
>
> **不要从第 1 行读 `kv_cache_manager.py`（922 行）或 `block_pool.py`（901 行）。**
> 按行号跳读，每处 20–40 行，读完就停。

### 2.1 必读（按顺序）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/core/kv_cache_utils.py:176` | `KVCacheBlock` | `block_id`/`ref_cnt`/`block_hash`/前驱后继；vLLM 叫 `prev_free_block`，nano 叫 `prev_free` | `pool`/`is_null` |
| 2 | `vllm/v1/core/kv_cache_utils.py:246` | `FreeKVCacheBlockQueue` | 为什么用侵入式双向链表（O(1) 中段删除）；fake head/tail | 构造函数的逐行链接 |
| 3 | `vllm/v1/core/kv_cache_utils.py:335` / `:379` / `:426` / `:447` | `popleft_n`/`remove`/`prepend_n`/`append_n` | 摘链后指针置 `None`；`remove` 只能删队列内块；队首=先驱逐，队尾=后驱逐 | 单块 `popleft` |
| 4 | `vllm/v1/core/block_pool.py:136` | `BlockPool` | `blocks`/`free_block_queue`/`cached_block_hash_to_block` 三个字段 | null block、events、metrics、`unpin_blocks` |
| 5 | `vllm/v1/core/block_pool.py:666` | `get_new_blocks()` | 先查余量、`popleft_n`、`ref_cnt += 1`；开缓存时先驱逐 | `_notify_reuse` |
| 6 | `vllm/v1/core/block_pool.py:752` / `:774` | `touch()`/`free_blocks()` | 命中块先摘出 free queue 再 `+1`；归零后按“有无 hash”分两段放回 | 多 pool 转发、events |
| 7 | `vllm/v1/core/kv_cache_manager.py:35` | `KVCacheBlocks` | 一个 `blocks` 列表 + `get_block_ids()` | 多 group 版本 |
| 8 | `vllm/v1/core/kv_cache_manager.py:131` | `KVCacheManager` | `req_to_blocks`、`block_pool`、`enable_caching` | coordinator/watermark/connector |
| 9 | `vllm/v1/core/kv_cache_manager.py:371` | `allocate_slots()` | **只读签名和 Blocks layout 注释**；主路径见 §5 T3 | 外部 KV、lookahead、reserved、sliding |
| 10 | `vllm/v1/core/kv_cache_manager.py:620` / `:782` | `free()`/`cache_blocks()` | “倒序释放，尾块先被驱逐”；`enable_caching=False` 时 `cache_blocks` 是 no-op | `remove_skipped_blocks`、缓存实现（M07） |
| 11 | `vllm/v1/worker/block_table.py:49` | `BlockTable` | CPU/GPU 两份表；`num_blocks_per_row` 为什么单独存 | PCP/DCP 字段 |
| 12 | `vllm/v1/worker/block_table.py:114` / `:127` / `:131` / `:137` | `append_row`/`add_row`/`clear_row`/`move_row` | 行内追加 vs 整行覆盖；清行为什么必须做 | — |
| 13 | `vllm/v1/worker/block_table.py:153` | `compute_slot_mapping()` | 签名 + 一句公式；式子在 `:374`：`slot = block_id * block_size + offset` | Triton kernel、CUDA graph padding |
| 14 | `vllm/v1/worker/gpu_worker.py:634` | `determine_available_memory()` | profiling 顺序：先测空闲，再减权重/激活，剩下给 KV | graph 内存估算 |
| 15 | `vllm/v1/worker/gpu_model_runner.py:7055` | `initialize_kv_cache()` | **只看骨架**：每层 `[num_blocks, block_size, kv_heads, head_dim]` | backend builder、kernel block size |
| 16 | `vllm/v1/kv_cache_interface.py:556` / `:578` | `page_size_bytes`/`FullAttentionSpec` | `page_size_bytes` 公式的来源 | sliding/DCP/padding |
| 17 | `vllm/v1/kv_cache_interface.py:1460` | `KVCacheConfig` | 多 group 长什么样；nano 只要单 group | 其余字段全跳 |

### 2.2 建议读（选读）与明确跳过

`vllm/v1/core/block_pool.py:868 · get_num_free_blocks()`（nano 改成 property）；
`vllm/v1/core/kv_cache_manager.py:723` / `:727 · get_blocks()/get_block_ids()`（调度器拿全量块的入口）；
`vllm/v1/core/sched/scheduler.py:1616 · _update_after_schedule()`（自增在 `:1629`，调度**之后**）。
**明确跳过**：coordinator 与 `single_type_kv_cache_manager.py`（多 group、CoW、SWA、
Mamba）；KV events/metrics/unpin；`allocate_slots()` 的 external/connector/reserved 参数；
partial block、fine-grained hash；prefix caching 真实逻辑（M07）。

---

## 3. 核心概念

### 3.1 block、block table、slot mapping

```text
物理块（block）：KV 显存的固定页，id 0..num_gpu_blocks-1
逻辑块号：请求内的第 j 块
  block_table[row] = [5, 9, 2, ...]      # 第 j 逻辑块 -> 物理块 5/9/2
slot：本步某 token 的写入地址
  slot = block_table[row][pos // B] * B + (pos % B)
```

一句话：**block 是页，block table 是页表，slot 是本步要写的那个地址。** block table
是请求级长期映射；slot mapping 是本步临时数组（`[num_actual_tokens]`），与 `positions` 对齐。

### 3.2 KV tensor layout（全课程定死这一种）

```python
k_cache: torch.Tensor  # [num_blocks, block_size, num_kv_heads, head_dim]
k_cache.view(-1, num_kv_heads, head_dim)[slot] = k[t]     # 写
```

对应 vLLM 的 `[num_blocks, block_size, num_kv_heads, head_dim]`。**不要改成
`[num_blocks, num_kv_heads, block_size, head_dim]`**：M04 的 Triton kernel 按前者
写 stride，改了两边一起错。

### 3.3 `ref_cnt` 状态机

```text
          get_new_blocks()            free_blocks()
  free ──────────────────► used ──────────────────► free
   ▲  ref=0                 ref=1                    ref=0
   │                          │  touch()（M07）
   └──────────────────────────┘  ref=2（两个请求共享）
```

块可被多个请求共享，所以 `ref_cnt` 是计数不是布尔；**只有 `ref_cnt == 0` 才能回 free
queue**；`ref_cnt > 0` 的块也可能还在 `cached` map 里（M07）；释放请求 = 对每个块
`ref_cnt -= 1`，不是清空。

### 3.4 free queue 的 LRU 顺序

队首 = 最久未用/最先驱逐。`free_blocks()` 把块分两段：有 `block_hash` 的放**队尾**
（保留更久，给后续请求复用），无 hash 的放**队首**（立即复用，GPU 局部性更好）。
M03 可以统一 `append_n` 到队尾（没缓存，两种块无区别），但**测试要断言“队首先出”**；
M07 补上两段策略，测试不用改。

### 3.5 失败返回 `None`；内部碎片

`BlockPool.get_new_blocks()` 余量不足抛 `ValueError`（调用方没先检查 = 程序员错误）；
`KVCacheManager.allocate_slots()` 余量不足**返回 `None`**（正常控制流，Scheduler 据此
抢占）。两个约定分清，否则 Scheduler 会把“显存不够”当崩溃。

分配用 `ceil(num_tokens / block_size)`，但 `num_full_blocks = num_tokens // block_size`。
例：20 token / 16 → 2 块，第二块只用 4 个 slot。block_size 越小碎片越少，block table
和 kernel 开销越大。

### 3.6 显存 sizing

```text
page_size_bytes = block_size * num_kv_heads * 2 * head_dim * dtype_size   # 每层每块
num_gpu_blocks  = available_kv_bytes // (num_layers * page_size_bytes)   # 全模型一块 = num_layers 份
```

Qwen3-0.6B、`block_size=16`、bf16：

```text
page_size_bytes = 16 * 8 * 2 * 128 * 2 = 65,536 B = 64 KiB（每层每块）
每块全模型     = 28 * 64 KiB = 1.75 MiB
4K 请求        = 4096 / 16 = 256 块
16GB 卡估算    = 0.85*16GiB - 1.2GB 权重 - ~2GiB 激活 ≈ 10.5 GiB
num_gpu_blocks ≈ 10.5 GiB / 1.75 MiB ≈ 6144 块 -> 约 24 个 4K 请求
```

24 是手算上界；真实值以 `determine_available_memory()` 输出为准。对比 §1 的“10 个
并发就 4.4 GiB”：同样显存能放两倍以上满载请求。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `prev_free_block`/`next_free_block` | `prev_free`/`next_free` | 改名（§6.4 已冻结） |
| `block_hash`（property，含 group id） | `block_hash: bytes` 或 `None` | 单 group |
| `get_cached_block(hash, group_ids) -> list` | `get_cached_block(hash) -> KVCacheBlock` 或 `None` | 单 group |
| `get_num_free_blocks()` | `num_free_blocks` property | 改名 |
| null block / KV events / metrics / unpin | 删除 | 与正确性无关，见 §7 坑 5 |
| coordinator（多 group） | `req_to_blocks: dict[str, list[KVCacheBlock]]` | 删除 |
| `get_computed_blocks -> (blocks, n, boundary)` | `(blocks, n)` | 删第三个返回值 |
| `BlockTable` 的 CPU/GPU 双 buffer + kernel | 先 CPU tensor，M04/M10 再上 GPU | 简化 |
| `compute_slot_mapping(...) -> None` | `-> torch.Tensor`（int64） | 便于单测 |
| `allocate_slots()` 13 个参数 | 4 个参数（§6.4） | 删 connector/spec/sliding |
| `cache_blocks()` 在 `allocate_slots()` 末尾 | M03 no-op；M07 改到 step 成功后 | 简化 |

**本模块新增的接口（§6 未冻结，M04/M07 依赖；改名先改 §6.4）**：

```python
class BlockPool:
    __init__(num_gpu_blocks, block_size, enable_caching=False)
    get_new_blocks(n) -> list[KVCacheBlock];  free_blocks(blocks);  touch(blocks)
    get_cached_block(hash) -> KVCacheBlock | None
    cache_full_blocks(request, blocks, num_cached, num_full, block_size)
    num_free_blocks -> int          # property

class BlockTable:
    __init__(block_size, max_num_reqs, max_num_blocks_per_req, device="cpu")
    add_row / append_row / clear_row / move_row
    compute_slot_mapping(query_start_loc, positions) -> torch.Tensor

reshape_and_cache(k, v, k_cache, v_cache, slot_mapping) -> None
```

`KVCacheBlocks`/`KVCacheManager` 严格用 §6.4，**不要加参数**。上面三个接口是 §6.4
没写的，按 vLLM 同名类删到最小，属于“本模块新增”。

---

## 5. 实现任务

> 按顺序做；每完成一个跑一次验收。**不要一次改 3 个文件。**
> 仓库现在没有 `nanovllm/core/` 和 `nanovllm/worker/`，T0 先建目录和失败测试。

### T0 · 建目录与失败测试（10 分钟）

新建 `nanovllm/core/{__init__,kv_cache_utils,block_pool,kv_cache_manager}.py`、
`nanovllm/worker/{__init__,block_table}.py`、`nanovllm/attention/ops/{__init__,cache}.py`、
`tests/test_m03_{block_pool,slot_mapping,gpu_kv,parity}.py`、`tests/kv_helpers.py`（§9）。
先写失败的测试（现在必然 `ImportError`）：

```python
# tests/test_m03_block_pool.py
from nanovllm.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue

def test_queue_is_fifo_and_append_goes_tail():
    q = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(4)])
    assert [b.block_id for b in q.popleft_n(2)] == [0, 1]
    q.append_n([KVCacheBlock(99)])       # 只测链表，不要求 99 属于池子
    assert [b.block_id for b in q.popleft_n(3)] == [2, 3, 99]
    assert len(q) == 0
```

验收：`python -m pytest tests/test_m03_block_pool.py -q` → 预期 2 failed/ImportError。
三个坑：① 空壳让测试假绿（至少 `raise NotImplementedError`）；② 测试写进 `nanovllm/`；
③ 忘了 `conftest.py` 已处理 `sys.path`，又去改 `PYTHONPATH`。

### T1 · `FreeKVCacheBlockQueue`（40 分钟）

文件：`nanovllm/core/kv_cache_utils.py`（签名见 §4）。要求侵入式双向链表、fake head/tail、
`popleft_n` 摘下的块指针置 `None`、`remove` 只能删队列内块、`__len__` 用计数器。

```text
popleft_n(n): assert n <= count; 从 head.next 摘 n 个; 首尾接回 fake head;
              每个 block.prev = block.next = None; count -= n
remove(b):    assert b.prev and b.next; b.prev.next = b.next; b.next.prev = b.prev;
              b.prev = b.next = None; count -= 1
```

验收：`python -m pytest tests/test_m03_block_pool.py -q -k queue`（5 passed）。
三个坑：① `remove` 刚 `popleft` 的块（指针全 `None`）应报错；② `append_n` 忘更新
`count`；③ fake head 的 `block_id=-1` 被当真实块返回。

### T2 · `BlockPool`（不开缓存，50 分钟）

文件：`nanovllm/core/block_pool.py`。

| 方法 | M03 行为 |
|---|---|
| `get_new_blocks(n)` | 余量不足抛 `ValueError`；`popleft_n`；`ref_cnt += 1`；开缓存时先驱逐（M07） |
| `free_blocks(bs)` | `ref_cnt -= 1`；归零后 `append_n` 队尾（M03 简化） |
| `touch(bs)` | `ref_cnt == 0` 时先 `remove`，再 `+1`（M07 用，M03 可先写） |
| `get_cached_block(h)` / `cache_full_blocks(...)` | M03 返回 `None` / 直接 `return`（占位） |
| `num_free_blocks` | `len(free_block_queue)` |

验收：`pytest tests/test_m03_block_pool.py -q -k "pool and not manager"`（6 passed）。
必须覆盖：分配后 `ref_cnt==1`、释放回队列、**100 轮分配/释放块数守恒**、超量抛
`ValueError`、队首先出。

三个坑：① 对 `ref_cnt==0` 的块再减成 -1；② 用 `list.remove()` 做中段删除（O(n)
且与侵入式链表混用断链）；③ `num_free_blocks` 用 `ref_cnt==0` 统计，把“在 cached
map 里但 ref=0”的块也算进去（M07 暴露）。

### T3 · `KVCacheManager`：分配与释放（50 分钟）

文件：`nanovllm/core/kv_cache_manager.py`。`KVCacheBlocks` 与 `KVCacheManager`
签名严格按 §6.4：`get_computed_blocks` 返回 `(KVCacheBlocks([]), 0)`（M03 空实现），
`get_block_ids(request_id)` 返回 `[b.block_id ...]`（不存在则 `[]`）。

**`allocate_slots()` 的 10 步（照抄，不要自己发明）**：

```text
 1. total_computed = request.num_computed_tokens + num_new_computed_tokens
 2. num_tokens_need = total_computed + num_new_tokens
 3. num_blocks_need = ceil(num_tokens_need / block_size)
 4. num_blocks_have = len(req_to_blocks[req_id])
 5. num_new_blocks = num_blocks_need - num_blocks_have
 6. 如果 num_new_blocks > block_pool.num_free_blocks: return None
 7. blocks = block_pool.get_new_blocks(num_new_blocks)
 8. 如果 new_computed_blocks 非空：
      - 把它们的 ref_cnt += 1（touch）
      - 追加到 req_to_blocks[req_id]
 9. req_to_blocks[req_id].extend(blocks)
10. return KVCacheBlocks(blocks)          # 只返回“本步新分配”的块
```

必须写进代码的四条：

1. 第 8 步 `touch` 只对命中块；M03 里 `new_computed_blocks` 恒 `None`。
2. 第 1 步读的是**调度前**的 `request.num_computed_tokens`。Scheduler 在
   `schedule()` 返回后、`_update_after_schedule()` 里才 `+= num_scheduled_tokens`
   （`vllm/v1/core/sched/scheduler.py:1616`，自增在 `:1629`）。用成调度后的值会
   每步多算一块。
3. 第 5 步可能为负：chunk 没填满已有块时 `num_new_blocks == 0`，返回空
   `KVCacheBlocks`，不是 `None`。
4. **先检查容量（第 6 步）、再 touch/追加（第 8 步）**；顺序反了，失败路径就会泄漏
   引用计数。`get_new_blocks` 抛异常是程序员错误，`None` 才是抢占信号。

`free()`：**倒序**（`reversed(req_to_blocks.pop(req_id))`，并清掉命中/缓存计数），
vLLM 在 `vllm/v1/core/single_type_kv_cache_manager.py:581 · free()` 也是
`free_blocks(reversed(...))`。

验收：`pytest tests/test_m03_block_pool.py -q -k manager`（5 passed）。断言：
20 token→2 块，32 token→2 块，33 token→3 块；池子只剩 1 块时要 2 块 → `None` 且
`req_to_blocks` 不变；free 后尾块先被拿走。

三个坑：① `num_blocks_need` 用 `//`（尾块不分配，写 KV 越界）；② `free` 没清
`req_to_blocks[req_id]`（抢占后 `num_blocks_have` 虚高）；③ 失败路径已经 touch
却没回滚（M07 经典 bug；M03 就要守住“先检查容量、后改状态”）。

### T4 · `BlockTable` + slot mapping（30 分钟）

文件：`nanovllm/worker/block_table.py`。表 `[max_num_reqs, max_num_blocks_per_req]`
int32，另存 `num_blocks_per_row: list[int]`。`append_row` 行尾追加；`add_row` 先清
再追加；`clear_row` 清前缀并归零；`move_row` 复制后**清空源行**（vLLM 注释：
防止 dummy batch 解引用旧 row）。

```text
for req i, for t in [qsl[i], qsl[i+1]):
    pos = positions[t]
    slot[t] = block_table[i][pos // B] * B + (pos % B)
```

验收：`pytest tests/test_m03_slot_mapping.py -q`（4 passed）。手算例子写进测试注释：

```text
B=4, block_table=[[7,3,11],[5,0,0]], qsl=[0,4,6], positions=[0,1,2,3,0,1]
req0: 28,29,30,31    req1: 20,21
```

边界：`pos=15/16` 跨块、prompt 非 16 倍数、两请求同 batch、`clear_row` 后无残留。

三个坑：① 用 `pos // B` 当物理块号（忘了查表）；② 把 `positions` 当 batch 内偏移
（它是**请求内绝对位置**，第二个请求从 0 重新开始）；③ 返回 int32（M04 要 int64）。

### T5 · KV tensor 分配 + `reshape_and_cache()`（40 分钟）

文件：`nanovllm/worker/gpu_worker.py`、`nanovllm/attention/ops/cache.py`。

```text
available_kv_bytes = total * gpu_memory_utilization - weights - activations - margin
num_gpu_blocks = int(available_kv_bytes // (L * page_size_bytes))   # L = num_layers = 28
k_caches = [torch.empty(num_gpu_blocks, B, H_kv, D, dtype=bf16, device="cuda") for _ in range(L)]
```

`reshape_and_cache` 用 index_copy，不写 Python for 循环：

```python
k_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slot_mapping, k)
```

验收：`pytest tests/test_m03_gpu_kv.py -q -m gpu`（2 passed）。覆盖：写已知 k/v
按 slot 读回 `torch.equal`；两个不同物理块同一 offset 不串；乱序 slot 也对。

三个坑：① 非连续 tensor 上 `.view` 报错；② cache 与 k 的 dtype/device 不一致；
③ `slot_mapping` 不是 int64，kernel 按 int32 读得到垃圾地址。

### T6 · 端到端对拍（30 分钟）

文件：`tests/test_m03_parity.py`。用 M02 的模型和 prompt 跑两条路径：A 连续 cache
直接 `k_cache[req][pos] = k`；B 分页 cache `reshape_and_cache`，先 gather 回连续布局
（仅对拍用，最终由 M04 替换）。

```python
torch.testing.assert_close(logits_paged, logits_contiguous, atol=1e-3, rtol=1e-3)
assert torch.equal(token_paged, token_contiguous)     # greedy
```

验收：`pytest tests/test_m03_parity.py -q -m gpu`（2 passed：prefill + 1 decode）。
三个坑：① gather 的块数算错，末尾读未初始化块 → NaN；② 忘了 attention 只能看
`seq_len` 以内（nano 无 null block）；③ 对拍没写 `atol/rtol`（bf16 至少 1e-3）。

### T7 · 收尾（10 分钟）

```bash
python -m pytest tests -q && python -m pytest tests -q -m gpu
git add nanovllm/core nanovllm/worker/block_table.py nanovllm/attention/ops tests
git commit -m "M03: paged KV cache (block pool / kv cache manager / block table)"
```

---

## 6. 测试与对拍

| 文件 | 层 | 关键断言 | 失败通常意味着 |
|---|---|---|---|
| `test_m03_block_pool.py` | CPU | 引用计数、100 轮守恒、LRU、`None` 语义 | 队列断链 / `ceil` 写错 |
| `test_m03_slot_mapping.py` | CPU | 手算 slot 相等、跨块、多请求 | 漏查表 / 位置用错 |
| `test_m03_gpu_kv.py` | GPU | 写入读回 `torch.equal`、块间不串 | layout / dtype 错 |
| `test_m03_parity.py` | GPU | logits `assert_close(atol=1e-3)` | slot 或 gather 错 |

**对拍基准**：M02 的连续 cache（不是 HF，也不是 vLLM）。固定 seed、固定 prompt、
greedy；先比 logits 再比 token。

| 现象 | 先查 |
|---|---|
| 第一步正常，第二步乱码 | decode 的 `positions`/slot 用了 batch 内偏移 |
| 输出逐渐重复 | 尾块 slot 越界，覆盖别的请求 |
| 块数每步 +1，很快 OOM | `allocate_slots` 用了调度后的 `num_computed_tokens` |
| attention 读到 NaN | block table 行没 `clear_row`，gather 到未初始化块 |

---

## 7. 常见坑

1. **`num_blocks_need` 用整除。** 20 token / 16 要 2 块；整除只给 1 块，写第 17 个 token 越界。
2. **`num_computed_tokens` 调度前/后搞反。** 自增在 `_update_after_schedule()`（`vllm/v1/core/sched/scheduler.py:1616`）；反了每步多算一块。
3. **释放没有倒序。** 头块先回队首、下次先复用头块，尾块（最新 KV）反而活最久，LRU 语义颠倒。
4. **`ref_cnt` 归零后没有清 `block_hash`。** M03 没事，M07 会命中“已被复用装新数据”的块；正确做法是**只在块被 `get_new_blocks` 拿走时驱逐 hash**（`vllm/v1/core/block_pool.py:729`），不是 free 时清（M07 §3.4）。
5. **padding 行用 0，而 0 是真实块。** vLLM 有 null block（`block_id=0` 占位），nano 不保留；`clear_row()` 必须真清行，attention 只读 `seq_len` 以内，M04 kernel 要用 `seq_len` mask。
6. **slot 用 `positions // block_size` 当物理块号。** 那只是逻辑块号，必须再查 `block_table[row][logical]`。
7. **KV layout 与 kernel 假设不一致。** 定死 `[num_blocks, block_size, num_kv_heads, head_dim]`；换 layout 要同时改 `reshape_and_cache`、M04 kernel 和测试。
8. **把 `get_new_blocks` 的 `ValueError` 当正常失败。** `BlockPool` 抛异常是调用方没先查余量；`KVCacheManager` 返回 `None` 才是容量不足，Scheduler 只处理 `None`。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m03_block_pool.py tests/test_m03_slot_mapping.py -q` 全绿；
- [ ] `pytest tests/test_m03_gpu_kv.py tests/test_m03_parity.py -q -m gpu` 全绿；
- [ ] 能白板画出 `allocate_slots` 的 10 步，并标出 `num_computed_tokens` 的读写时机；
- [ ] 能手算 Qwen3-0.6B 的 `page_size_bytes`（64 KiB/层/块）和 4K 请求块数（256）；
- [ ] 倒序 free、`clear_row`、int64 slot 三处有专门测试；100 轮分配/释放块数守恒；
- [ ] `PROGRESS.md` 写清“无 null block / 单 group / 未开缓存”；
- [ ] commit：`M03: paged KV cache (block pool / kv cache manager / block table)`。

---

## 9. 交接：给 M04 和 M07

**新增文件**（M04/M06/M07 会 import，不要改名）：

```text
nanovllm/core/{kv_cache_utils,block_pool,kv_cache_manager}.py
nanovllm/worker/block_table.py        BlockTable
nanovllm/attention/ops/cache.py       reshape_and_cache
tests/kv_helpers.py                   M07 复用
```

`tests/kv_helpers.py` 只放三个函数：`make_pool(num_blocks, block_size=16,
enable_caching=False)`、`assert_block_conservation(pool, expected_total)`、`slot_of(...)`。

**公开 API**：§4 的三段新增接口 + §6.4 冻结的 `KVCacheManager`/`KVCacheBlocks`；
`allocate_slots` 只返回本步新块，全量块用 `get_block_ids(request_id)`。

**已知限制**：单 KV group、全模型统一 `block_size`；**无 null block**（attention 必须
按 `seq_len` mask）；**不开前缀缓存**（`get_cached_block` 恒 `None`）；**不支持
partial block / CoW**（M07 也只做整块命中）；`BlockTable` 先 CPU。

**M04 入口**：读 `docs/M04_paged_attention.md`。`AttentionMetadata` 已由 §6.5 冻结：
`block_table [num_reqs, max_num_blocks] int32`、`slot_mapping [num_actual_tokens] int64`、
`seq_lens [num_reqs] int32`。M04 的 `forward()` 先 `reshape_and_cache` 再 paged attention。
**M07 入口**：读 `docs/M07_prefix_caching.md`；只改 `BlockPool` 的 `get_cached_block`/
`cache_full_blocks`/驱逐分支，以及 `KVCacheManager` 的 `get_computed_blocks`/
`allocate_slots` 第 1 步，**不改签名**。

`PROGRESS.md` 追加：M03 完成 `KVCacheBlock`/`FreeKVCacheBlockQueue`/`BlockPool`
（`enable_caching=False`）、`KVCacheManager`、`BlockTable`+slot mapping、
`reshape_and_cache`；已知偏差为单 group、无 null block、无 partial/CoW、前缀缓存未启用。

---

## 10. 自测题

1. block table 和 slot mapping 的区别？各解决什么问题？
2. 为什么 `allocate_slots` 失败要返回 `None` 而不是抛异常？
3. `ref_cnt` 在什么时机 +1 / -1？举出三个调用点。
4. 为什么释放请求时要倒序释放块？顺序释放会让 LRU 变成什么样？
5. Qwen3-0.6B 的 `page_size_bytes` 是多少？16GB 卡大约能放多少块、多少个 4K 请求？

---

## 11. 教练流程（给 agent 用，约 4 小时）

1. **20 分钟**：让用户先算 §1 的显存账和 §3.6 的 `page_size_bytes`；算错就停。
2. **40 分钟**：带读 §2.1 的 1–6（队列 + BlockPool），只读指定行；禁止打开文件全文。
3. **50 分钟**：用户写 T1/T2；先让队列 FIFO 测试绿，再写 100 轮守恒；卡住只提示
   “哪条指针没接上”，不给实现。
4. **50 分钟**：用户写 T3；先在纸上跑通 10 步的第 2/4/6 步；重点检查“先检查容量、后改状态”。
5. **30 分钟**：T4 slot mapping；让用户手算 §5 T4 的例子并解释 `pos=15/16`。
6. **30 分钟**：T5/T6 GPU 测试；先 `torch.equal` 读回，再对拍 logits。
7. **15 分钟**：自测题 + 更新 `PROGRESS.md` + commit。

**禁止**：让用户从第 1 行读 900 行文件；一上来贴完整实现或替用户写
`allocate_slots`；跳过 block pool 单测直接写 manager；让用户安装 vLLM 本体跑对比。
