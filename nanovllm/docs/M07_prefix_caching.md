# M07 · Prefix Caching：链式哈希、命中、LRU 驱逐

> **模块目标**：打开 `enable_prefix_caching`，让相同前缀的请求复用已算好的 KV block：
> 链式哈希 → 查表命中 → `touch` 引用 → 尾部重算 → 释放后仍可命中 → 新块挤占时驱逐。
>
> **预计工时**：3–4 小时　**依赖**：M03（`BlockPool`/`KVCacheManager`）、M06（step 边界）
> **测试**：CPU（哈希/命中/驱逐）+ GPU（端到端一致性与 TTFT）
>
> **产出**：`tests/test_m07_prefix_cache.py` 全绿；`benchmarks/bench_prefix_cache.py`
> 输出命中率与 TTFT；缓存开/关的 greedy 输出逐 token 一致。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m07_prefix_cache.py -q          # 23 passed
python -m pytest tests/test_m07_prefix_cache.py -q -m gpu   # 2 passed
python benchmarks/bench_prefix_cache.py --num-reqs 8 --prefix-len 512
# 期望：第 2 个请求起 hit_tokens=496/512；TTFT 明显低于第 1 个
```

以及五个能脱口而出的答案：

1. 块哈希为什么必须带 parent？举一个不带就会错的例子。
2. 为什么命中长度要限制在 `num_tokens - 1`？
3. `ref_cnt == 0` 的块为什么还能被命中？
4. `free_blocks` 为什么把有 hash 的块放队尾？
5. 两个请求共享一个块，什么时候这块显存才能真正被复用？

---

## 1. 为什么需要这个模块（10 分钟）

真实负载里大量请求共享同一段前缀：system prompt、few-shot 示例、RAG 的固定指令。
没有前缀缓存时，每个请求都要把这 512 个 token 重新 prefill 一遍——KV 一样、算力白花。

```text
8 个请求，共享 512-token system prompt，B=16
不开缓存：8 × 512 = 4096 个 token 的 prefill
开缓存：  512（第一个）+ 7 × 16（尾部对齐重算）≈ 624
省掉约 85% 的 prefill 计算，TTFT 主要就是 prefill 时间
```

省算力的同时还能省显存：第二个请求的 31 个命中块与第一个请求**指向同一批物理块**，
`ref_cnt` 变 2，而不是复制一份 KV。M03 已经把“块可被引用”准备好了，M07 只是把
“按内容找到块”这件事接上。

M03 的接口**一个都不改**：`enable_caching` 从 `False` 变 `True`，`get_computed_blocks`
从“返回空”变成真查表，`cache_blocks` 从 no-op 变成写 hash map。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`
> **不要通读** `kv_cache_manager.py`（922 行）、`block_pool.py`（901 行）或
> `scheduler.py`（3436 行）。只读下表行号前后 20–40 行。

### 2.1 必读（按顺序）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/request.py:230` | `update_block_hashes()`（构造时调用） | prompt 的满块在构造时就算好哈希 | 多模态 extra keys |
| 2 | `vllm/v1/request.py:285` | `update_block_hashes()` | 只 `extend` hasher 的返回值；增量、幂等 | — |
| 3 | `vllm/v1/core/kv_cache_utils.py:118` | `NONE_HASH` | 根哈希是非空常量；nano 用 `sha256("nanovllm-none-hash")` | seed 随机化 |
| 4 | `vllm/v1/core/kv_cache_utils.py:695` | `hash_block_tokens()` | `hash_fn((parent, tuple(tokens), extra_keys))`；parent 为 `None` 换 `NONE_HASH` | `extra_keys` 的真实来源 |
| 5 | `vllm/v1/core/kv_cache_utils.py:941` | `get_request_block_hasher()` | 起点 `len(block_hashes)`；`start + B > num_tokens` 就停 | MM/LoRA 分支 |
| 6 | `vllm/v1/core/block_pool.py:35` / `:62` | `BlockHashToBlockMap` / `get_one_block()` | hash → 块的反查结构；nano 用普通 dict | 多 group/重复块逻辑 |
| 7 | `vllm/v1/core/block_pool.py:198` | `get_cached_block()` | 命中返回块，未命中返回 `None`；单 group 只查一次 | `kv_cache_group_ids` 循环 |
| 8 | `vllm/v1/core/block_pool.py:226` | `cache_full_blocks()` | `blocks[num_cached:num_full]` 逐块 `set_block_hash` + 插入 map | events、block_mask、partial 提升断言 |
| 9 | `vllm/v1/core/block_pool.py:729` | `_maybe_evict_cached_block()` | 先从 map 删条目、再 `reset_hash()`；返回是否驱逐 | metrics、events |
| 10 | `vllm/v1/core/block_pool.py:752` | `touch()` | `ref_cnt == 0` 先从 free queue 摘掉，再 `+1` | metrics |
| 11 | `vllm/v1/core/block_pool.py:774` | `free_blocks()` | 有 hash 的放队尾、无 hash 的放队首；注释写明 LIFO/FIFO 原因 | 多 pool 转发 |
| 12 | `vllm/v1/core/kv_cache_manager.py:264` | `get_computed_blocks()` | `max_cache_hit_length = num_tokens - 1`；命中长度按块下取整 | connector、shared_prefix_boundary |
| 13 | `vllm/v1/core/single_type_kv_cache_manager.py:733` | `FullAttentionManager.find_longest_cache_hit()` | 从第 0 块顺序查，**缺一块就 break**（链式哈希的性质） | fine-grained、SWA、CoW |
| 14 | `vllm/v1/core/single_type_kv_cache_manager.py:267` / `:304` | `add_local_computed_blocks()` / `block_pool.touch()` | **touch 发生在 `allocate_slots` 里，不在 `get_computed_blocks` 里** | external 块、null 填充 |
| 15 | `vllm/v1/core/kv_cache_manager.py:371` / `:616` | `allocate_slots()` / `cache_blocks()` 调用点 | vLLM 在分配末尾就 cache（此时 KV 还没算） | delay_cache、external、watermark |
| 16 | `vllm/v1/core/sched/scheduler.py:1037` | waiting 循环里的 `get_computed_blocks(request)` | 命中结果怎么变成 `num_computed_tokens` | connector 分支 |
| 17 | `vllm/v1/core/sched/scheduler.py:1616` | `_update_after_schedule()` | 调度后才 `num_computed_tokens += n`（`:1629`），与命中值相加 | 其余字段 |
| 18 | `vllm/v1/core/kv_cache_manager.py:782` | `cache_blocks()` | `num_full = num_tokens // block_size`；只缓存满块 | 多 group 实现 |

### 2.2 建议读与明确跳过

建议读 `vllm/v1/core/single_type_kv_cache_manager.py:581 · free()`（倒序释放）与
`:476 · cache_blocks()`（`num_cached_block` 计数器）。**明确跳过**：partial block / CoW
（`vllm/v1/core/kv_cache_utils.py:241 · KVCacheBlockCopy`、`_partial_hit_reqs`）、
fine-grained hash、KV events、connector、多 group、SWA/Mamba 的 pinning。

---

## 3. 核心概念

### 3.1 链式哈希

```text
block[0] = H(NONE_HASH,        tokens[0:B])
block[1] = H(block[0].hash,    tokens[B:2B])
block[2] = H(block[1].hash,    tokens[2B:3B])
```

反例：`[A][B]` 与 `[C][B]`，第二块 token 都是 B。若哈希只算 `tokens`，两者同 hash，
第二个请求会读到错误前缀的 KV。带上 parent 后 `H(A,B) != H(C,B)`。

nano 的 `hash_block_tokens(hash_function, parent, tokens, extra_keys=None)` 与 vLLM
同构；`parent=None` 在函数内部换成 `NONE_HASH`（M00 已实现）。

### 3.2 只哈希满块

```text
num_full_blocks = request.num_tokens // B      # 不是 ceil，也不是 num_computed
```

prompt 10 token、B=4 → 只哈希 2 块，尾部 2 个 token 的块还会继续长。等 decode 把它
填满（或 chunked prefill 再算 2 个），`append_output_token_ids()` 会触发
`update_block_hashes()`，第 3 块才进哈希表。

### 3.3 命中上限 `num_tokens - 1`

全命中时一个 token 都没算，**没有 logits**。所以：

```text
max_cache_hit_length = request.num_tokens - 1
max_blocks           = max_cache_hit_length // block_size
```

prompt 8、B=4：上限 7 → 命中 1 块 = 4 token，必须重算 4 个 token。
prompt 5：上限 4 → 命中 1 块，重算 1 个 token。
因为要按块对齐，实际可能多算最多 `B-1` 个 token——这是正确性换来的代价。

### 3.4 `ref_cnt` 与 cache 是两件事

| `ref_cnt` | 有 `block_hash` | 在 free queue | 含义 |
|---|---|---|---|
| >0 | 有 | 否 | 正在被一个或多个请求使用，也可能被命中 |
| 0 | 有 | 是 | 空闲但**仍可命中**；只有被 `get_new_blocks` 拿走才驱逐 |
| 0 | 无 | 是 | 纯空闲块，无复用价值 |

关键：块进 free queue ≠ 立刻失效。`get_new_blocks` 弹出块后先
`_maybe_evict_cached_block()`，从 map 删掉旧 hash，再 `ref_cnt += 1`。这样“命中”与
“复用装新数据”互斥。

### 3.5 LRU 两段

`free_blocks` 把有 hash 的块 `append_n` 到队尾（慢点被复用），无 hash 的
`prepend_n` 到队首（尽快复用）。于是队首永远是最该被驱逐的；prefix cache 的块在
队列里“活得久”，命中率更高。M03 的统一队尾是简化版，M07 要改成分两段。

### 3.6 cache 写入时机

vLLM 在 `allocate_slots()` 末尾就调用 `cache_blocks()`（`vllm/v1/core/kv_cache_manager.py:616`），
此时这一步的 KV 还没写进显存。它安全的前提是：调度与执行在同一进程、顺序执行，
中间没有第二个请求能查到这些 hash。

nano 选择**step 成功之后再 cache**（KV 写完、采样成功）：

```text
schedule() -> execute_model() -> sample_tokens() -> 成功
          -> manager.cache_blocks(req, req.num_computed_tokens)   # 这里
```

好处是异常/abort 不会留下“hash 已发布但 KV 是垃圾”的块。若 M06 已经照 vLLM 写在
`allocate_slots` 里，**删掉那处调用**，只保留 step 成功后这一处；同步执行下两者都
不会错，但“先写后发布”更容易在 M09 异步化时守住。

### 3.7 不支持 partial block / COW

vLLM 0.32 支持“命中块的一部分”并在写入时做 copy-on-write
（`KVCacheBlockCopy`、`_partial_hit_reqs`）。nano **只做整块命中**：

- prompt 6 token、B=4 → 最多命中 4 token，尾部 2 个一定重算；
- 不会出现“共享块被两个请求同时写”的情况，因此**不需要 CoW**；
- 命中率对不齐块边界的 prompt 会低一些，这是有意的简化。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `BlockHashWithGroupId`（hash + group id） | `bytes` | 单 group，去掉 group id |
| `BlockHashToBlockMap`（`vllm/v1/core/block_pool.py:35`） | `dict[bytes, KVCacheBlock]` | 简化；一个 hash 只对应一块 |
| `get_cached_block(hash, group_ids) -> list` | `get_cached_block(hash) -> KVCacheBlock` 或 `None` | 单 group |
| `find_longest_cache_hit()` 类方法 | `get_computed_blocks()` 里的 15 行循环 | 删除继承树 |
| `allocate_new_computed_blocks()` 两阶段 + 多 group | `touch` + `extend` | 删除 |
| `get_computed_blocks -> (blocks, n, boundary)` | `(blocks, n)` | §6.4 冻结 |
| partial hit + CoW | **删除** | 只做整块命中 |
| `cache_blocks()` 在 `allocate_slots()` 内 | step 成功后调用 | 简化，见 §3.6 |
| `PrefixCacheStats` | `hit_*` 在 manager、`evictions` 在 pool | 诊断用，非公共 API |
| `enable_caching` | 来自 `CacheConfig.enable_prefix_caching` | 同一个开关 |

**接口不变**：`KVCacheBlocks` / `KVCacheManager` 严格按 `00_architecture.md` §6.4，
不改签名。新增的只有 `BlockPool` 的两个方法体（M03 已占位）和 `KVCacheManager`
内部的 `num_cached_block` / `stats`；`BlockPool.get_cached_block(hash)` 的签名沿用
M03 的“本模块新增”版本。

---

## 5. 实现任务

> M03 的测试必须先是绿的。每完成一个任务跑一次验收，不要一次改 4 个文件。

### T1 · 打开开关，接上 hasher（15 分钟）

文件：`nanovllm/engine/core.py`（或 M06 组装 `Request` 的地方）、`nanovllm/config.py`。

```python
# 组装 Request 时：
block_hasher = get_request_block_hasher(block_size, hash_fn) if enable_prefix_caching else None
Request(..., block_size=block_size, block_hasher=block_hasher)
# KVCacheManager:
KVCacheManager(block_pool, block_size, enable_caching=config.cache_config.enable_prefix_caching)
```

`hash_fn` 用 `hashlib.sha256(repr(x).encode()).digest()`；**不要**用内置 `hash()`（进程内随机）。

验收：`pytest tests/test_m07_prefix_cache.py -q -k "toggle or hasher"`（3 passed）。
三个坑：① `Request` 构造后 `block_hashes` 为空（hasher 没传）；② `block_size` 两处
不一致（§6.1 要求只留一个来源）；③ 开关关了但 `BlockPool` 仍建 `cached` map。

### T2 · `BlockPool.get_cached_block` + `cached` map（25 分钟）

文件：`nanovllm/core/block_pool.py`。

```text
get_cached_block(h):
    if not enable_caching: return None
    return self.cached.get(h)

cache_full_blocks(request, blocks, num_cached, num_full, B):
    if not enable_caching or num_cached >= num_full: return
    assert num_full <= len(request.block_hashes)
    for i in range(num_cached, num_full):
        b, h = blocks[i], request.block_hashes[i]
        assert b.block_hash is None            # 命中块不会被重复 cache
        if h in self.cached: continue          # 同内容块已登记；一个 hash 只登记一块
        b.block_hash = h; self.cached[h] = b
```

验收：`pytest ... -q -k "cache_lookup or cache_full"`（4 passed）。断言：写入后
`cached[h] is block`、`block.block_hash == h`；`num_cached==num_full` 时幂等；关缓存
时 `cached` 永远为空。

三个坑：① 用 `blocks[i]` 而 `i` 从 0 开始（应从 `num_cached` 偏移）；② 把
`num_full` 当成 token 数；③ 忘记 `assert b.block_hash is None`，命中块被二次插入。

### T3 · `get_computed_blocks()`（35 分钟）

文件：`nanovllm/core/kv_cache_manager.py`。

```text
def get_computed_blocks(request):
    if not enable_caching: return KVCacheBlocks([]), 0
    max_hit_tokens   = request.num_tokens - 1          # §3.3
    max_hit_blocks   = max_hit_tokens // block_size
    hit = []
    for h in request.block_hashes[:max_hit_blocks]:
        b = block_pool.get_cached_block(h)
        if b is None: break                            # 链式：缺一块后面全缺
        hit.append(b)
    tokens = len(hit) * block_size
    stats.hit_tokens += tokens; stats.hit_requests += bool(hit)
    return KVCacheBlocks(hit), tokens
```

**这里不要 `touch`**：touch 在 `allocate_slots` 里做（vLLM 的
`vllm/v1/core/single_type_kv_cache_manager.py:304`）；先查后分配可能失败，提前 touch 会泄漏引用。

验收：`pytest ... -q -k computed_blocks`（4 passed）。断言：共享 8 token → 8；分叉
→ 0；尾块 6 token → 4；完全相同 8 token → **4**（不是 8）。

三个坑：① 忘了 `-1`，全命中时没有 logits；② 用 `num_tokens // B` 当上限（应先减 1）；
③ 命中后没按块边界取整，返回非 B 倍数。

### T4 · `allocate_slots()` 命中路径（30 分钟）

文件：`nanovllm/core/kv_cache_manager.py`，在 M03 的 10 步第 8 步里补：

```text
if new_computed_blocks is not None and len(new_computed_blocks.blocks) > 0:
    block_pool.touch(new_computed_blocks.blocks)     # ref_cnt++，必要时移出 free queue
    req_to_blocks[rid].extend(new_computed_blocks.blocks)
    num_cached_block[rid] = len(new_computed_blocks.blocks)
```

`touch` 放容量检查前还是后？M03 的 10 步已经定成**先检查容量（第 6 步）、再 touch/追加
（第 8 步）**，和 vLLM 0.32 一致（先 `required > available`，再
`allocate_new_computed_blocks()`，见 `vllm/v1/core/kv_cache_manager.py:371`），失败路径天然干净。
如果你在 M03 里把顺序写反了，M07 这里必须补回滚测试（`ref_cnt -= 1`、截断
`req_to_blocks`）。

验收：`pytest ... -q -k "allocate_hit"`（3 passed）。断言：命中 2 块后 `ref_cnt` 加 2；
若原块已在 free queue，free 数减 2；`get_block_ids(req)` 前两块是命中块；失败时
`ref_cnt` 与 `req_to_blocks` 都不变（按上一条：要么先检查，要么回滚）。

三个坑：① 命中块没 touch，另一个请求 free 后块被回收；② touch 了却没追加到
`req_to_blocks`，后续 `cache_blocks` 对不上号；③ `num_cached_block` 没设，导致
`cache_full_blocks` 从 0 开始重复插入命中块（触发 `assert h not in cached`）。

### T5 · `cache_full_blocks()` 的调用时机（35 分钟）

文件：`nanovllm/core/kv_cache_manager.py` + `nanovllm/engine/core.py`（step 边界）。

```text
def cache_blocks(request, num_computed_tokens):
    if not enable_caching: return
    num_full = min(num_computed_tokens // block_size, len(request.block_hashes))
    num_cached = num_cached_block.get(rid, 0)
    if num_full <= num_cached: return
    block_pool.cache_full_blocks(request, req_to_blocks[rid], num_cached, num_full, block_size)
    num_cached_block[rid] = num_full
```

调用点：`execute_model()` + `sample_tokens()` 成功后，对每个被调度请求调用 `cache_blocks(req, req.num_computed_tokens)`。

验收：`pytest ... -q -k "cache_after_step or chunked"`（3 passed）。断言：prefill 8
token 后 `cached` 有 2 项；再 decode 到 9 token 仍是 2 项（尾块未满）；到 12 token
变 3 项；命中块不会被重新 cache。

三个坑：① 用 `request.num_tokens` 而非传入的 `num_computed_tokens`；② `allocate_slots`
里也 cache 了却没删，step 失败留下脏 hash；③ `num_full` 没和 `len(block_hashes)` 取 min。

### T6 · 驱逐与 LRU（35 分钟）

文件：`nanovllm/core/block_pool.py`。

```text
get_new_blocks(n):
    blocks = free_queue.popleft_n(n)
    for b in blocks:
        if enable_caching: _maybe_evict_cached_block(b)   # 先删 map + 清 hash
        assert b.ref_cnt == 0
        b.ref_cnt += 1
    return blocks

_maybe_evict_cached_block(b):
    if b.block_hash is None: return False
    cached.pop(b.block_hash, None); b.block_hash = None; self.evictions += 1; return True

free_blocks(ordered):                 # 调用方必须传 reversed(req_blocks)
    for b in ordered:
        b.ref_cnt -= 1
        if b.ref_cnt == 0:
            (front if b.block_hash is None else back).append(b)
    free_queue.prepend_n(front); free_queue.append_n(back)
```

验收：`pytest ... -q -k "evict or lru"`（4 passed）。断言：池子 6 块、A 占满后释放，
B 要 6 块 → A 的 6 个 hash 全部从 `cached` 消失；C 用 A 的前缀 → 命中 0；有 hash 的
块在队尾（再分配时最后才被拿走）。

三个坑：① 只摘 free queue、忘了删 `cached` → 命中“已装新数据”的块；② 清了
`block_hash` 却没删 map（反向同样错）；③ `free_blocks` 两段写反，命中块立刻被复用。

### T7 · 指标（15 分钟）

文件：`nanovllm/core/kv_cache_manager.py`。

```python
self.stats = {"hit_tokens": 0, "hit_requests": 0}   # evictions 在 BlockPool.evictions
```

在 `get_computed_blocks` 里累加 `hit_tokens`/`hit_requests`；`BlockPool.__init__` 建
`self.evictions = 0`，manager 读 `block_pool.evictions`。stats 不作公共 API。

验收：`pytest ... -q -k stats`（2 passed）：跑两个共享前缀请求后
`hit_tokens == 8`、`hit_requests == 1`。

三个坑：① 把“命中 token”算成“命中块数”；② 每个 step 都重复累加同一请求的命中；
③ 驱逐计数在 `block_hash is None` 时也加。

### T8 · 测试与 benchmark（40 分钟）

文件：`tests/test_m07_prefix_cache.py`、`benchmarks/bench_prefix_cache.py`。

benchmark 骨架（命令行参数自己定，固定 seed）：

```text
for i in range(num_reqs):
    add request(prompt = shared_prefix + suffix_i, max_tokens=1)
    t0 = time(); step() 直到该请求完成; ttft[i] = t1 - t0
print(hit_tokens, hit_rate, ttft[0], median(ttft[1:]))
```

验收：`python -m pytest tests/test_m07_prefix_cache.py -q`，以及
```bash
python benchmarks/bench_prefix_cache.py --num-reqs 8 --prefix-len 512
```

三个坑：① 8 个请求一次性 add，第一个还没 cache 完就都进来了，命中率接近 0；
② benchmark 没固定 `max_tokens`，TTFT 被 decode 污染；③ 只报“快了”不报命中 token 数。

---

## 6. 测试与对拍

统一用 `block_size=4`，greedy，`max_tokens=1`，缓存开/关各跑一遍比对输出。测试复用
M03 的 `tests/kv_helpers.py`（`make_pool` / `assert_block_conservation`），不要复制 pool 构造。

| 场景 | 构造（B=4） | 断言 |
|---|---|---|
| 共享前缀 | A `[1..8]`；B `[1..8,201,202,203]` | B `get_computed_blocks` 返回 8；B 首步 `num_scheduled_tokens=3`；输出与关缓存一致 |
| 前缀分叉 | A `[1,2,3,4,5,6,7,8]`；C `[9,10,11,12,5,6,7,8]` | C 命中 0；`A.block_hashes[1] != C.block_hashes[1]`（token 相同、parent 不同） |
| 尾块 | A、B 都是 `[1..6]` | B 命中 4（不是 6）；`len(B.block_hashes)==1` |
| 全命中 | A、B 都是 `[1..8]` | B 命中 4（`(8-1)//4=1` 块）；首步至少重算 4 token；logits 来自重算 |
| 驱逐 | 池 6 块；A `[1..24]` 占满后释放；B 用另一段 24 token | B 分配后 A 的 hash 不在 `cached`；C 用 A 前缀命中 0；C 输出 == 全新引擎输出 |
| 引用计数 | A、B 共享 2 块；先 free A | 共享块 `ref_cnt==1` 且不在 free queue；B 继续 decode 输出不变；free B 后 `ref_cnt==0` |

另外两条：

- **一致性**：同一组 prompt，缓存开/关的 greedy 序列必须逐 token 相等
  （`assert out_on == out_off`）。这是最重要的正确性断言。
- **性能**：8 个相同 512-token prompt，B=16，`max_tokens=1`。第 2–8 个请求的
  `max_hit = 511//16 = 31` 块 = 496 token；断言 `stats["hit_tokens"] == 7 * 496`，
  TTFT 只记录、不做硬断言（共享 GPU 上会抖）。

失败现象 → 原因：

| 现象 | 先查 |
|---|---|
| 第二个请求输出与第一个不同 | 命中了脏块 / 哈希没带 parent |
| 全命中请求 logits 全 0 或 shape 空 | 忘了 `num_tokens - 1` |
| 命中后块被别的请求复用，输出渐变乱码 | 命中没 `touch` |
| `cached` 里残留已复用块的 hash | 驱逐只动了队列没动 map |
| 相同前缀但命中 0 | 尾块被哈希 / `num_tokens` 与 `num_computed_tokens` 混用 |

---

## 7. 常见坑

1. **哈希了尾块。** 尾块会继续长，命中后拿到“只算了一半”的 KV；只哈希 `num_tokens // B` 个满块。
2. **没有 `num_tokens - 1`。** 全命中时没有 token 被算，logits 是旧的/空的；至少重算一个块。
3. **命中块没有 `touch`。** 块还在 free queue 里，另一个请求分配时把它驱逐，正在读它的请求读到新数据。
4. **`free_blocks` 把有 hash 的块放错队首。** 命中块立刻被复用；有 hash → 队尾，无 hash → 队首。
5. **驱逐只从 free queue 移除，忘了从 `cached` 删。** 下次命中一个已装别人 KV 的块，输出错乱且极难复现。
6. **多个请求共享块，`free` 时直接清 hash。** 另一个请求还在用；hash 只在 `get_new_blocks` 驱逐时清。
7. **`cache_blocks` 在 KV 写入前调用且执行不同步。** 发布未初始化 KV；nano 放到 step 成功后（§3.6）。
8. **`num_cached_block` 不维护。** 每次从 0 开始，重复插入命中块，触发 `assert h not in cached`。
9. **同一 hash 登记两个块。** 重复内容块（如全命中后重算的尾块）再次 cache 时覆盖 map，驱逐会误删别人的映射；`h in cached` 时跳过。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m07_prefix_cache.py -q` 全绿（含共享/分叉/尾块/全命中/驱逐）；
- [ ] 缓存开/关的输出逐 token 一致；
- [ ] 两个请求共享块时，free 一个不影响另一个；两个都 free 后块回 free queue；
- [ ] 驱逐后 `cached` 无残留 hash，且不会命中脏块；
- [ ] `bench_prefix_cache.py` 能打印 hit tokens / hit rate / TTFT，并解释第 2 个请求
      为什么是 496 而不是 512；
- [ ] `PROGRESS.md` 写清“只支持 full block、无 CoW、cache 在 step 成功后”；
- [ ] commit：`M07: prefix caching (chained hash / LRU eviction / refcount reuse)`。

---

## 9. 交接：给 M08

**修改文件**：`nanovllm/core/block_pool.py`（`cached`、`cache_full_blocks`、驱逐、
`free_blocks` 两段）、`nanovllm/core/kv_cache_manager.py`（`get_computed_blocks`、
`allocate_slots` 第 1 步、`cache_blocks`、`num_cached_block`、`stats`）、
`nanovllm/engine/core.py`（step 成功后调用 `cache_blocks`）、`benchmarks/bench_prefix_cache.py`。

**公开 API**：不变（§6.4）。`BlockPool` 的 `get_cached_block(hash)` /
`cache_full_blocks(request, blocks, num_cached, num_full, block_size)` 沿用 M03。
`KVCacheManager.stats` 是诊断字段，M08 可以读，但不要写。

**已知限制**：只做整块命中（无 partial block / CoW）；单 group；哈希不跨进程持久化；
cache 放在 step 成功后（同步前提）；`block_size` 必须与哈希粒度一致。

**M08 入口**：`docs/M08_sampling.md`。采样与 detokenize 不依赖前缀缓存，但要注意：
同一 prompt 开/关缓存必须给出相同 token；M08 的 logprobs 测试要关缓存跑，避免
“重算块”影响数值比较。

`PROGRESS.md` 追加：M07 完成链式哈希、`get_computed_blocks`（`num_tokens-1`）、
`allocate_slots` touch、step 成功后 `cache_full_blocks`、LRU 两段 free、驱逐、stats；
已知偏差为只支持 full block、哈希不持久化、cache 时机与 vLLM 不同；M08 入口为采样/输出。

---

## 10. 自测题

1. 为什么块哈希要带 parent？举一个不带就会命中的反例。
2. 为什么 `get_computed_blocks` 要把命中长度限制在 `num_tokens - 1`？
3. `ref_cnt == 0` 的块为什么还能被命中？
4. `free_blocks` 为什么把有 hash 的块放队尾？
5. 两个请求共享一个块，什么时候这块显存才能真正被复用？

---

## 11. 教练流程（给 agent 用，约 3.5 小时）

1. **20 分钟**：让用户复述 §3.1–§3.3，并在纸上算“8 token、B=4、全命中”能命中几个。
2. **40 分钟**：带读 §2.1 的 1–8（哈希 + 查表），只读指定行；禁止打开 900 行文件。
3. **40 分钟**：用户写 T1–T3；先让“共享 8 token 命中 8”绿，再补分叉与尾块。
4. **50 分钟**：用户写 T4–T6；重点检查“失败路径不改 ref_cnt”“驱逐先删 map 再清 hash”。
5. **40 分钟**：T7/T8 + benchmark；先断言 `7*496`，再看 TTFT 表。
6. **15 分钟**：自测题 + 更新 `PROGRESS.md` + commit。

**禁止**：一上来做驱逐（先把“两个请求共享前缀”跑通）；替用户写 `get_computed_blocks`；
让用户安装 vLLM 或通读 `block_pool.py`；把 partial block/CoW 加进 nano（明确不做）。
