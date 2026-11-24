# M00 · 全景与契约层：`Request` / `SchedulerOutput` / 块哈希

> **模块目标**：先不碰 GPU，把 V1 的“三层数据契约”亲手写一遍。
> 做完之后，你看任何 vLLM 调度代码都能回答三个问题：
> 这一步谁改了什么？跨进程传的是什么？为什么这样设计？
>
> **预计工时**：2–3 小时　**依赖**：无　**测试**：纯 CPU
>
> **产出**：`pytest tests -q` 从 failed 变全绿；一张手推的调度时序图。

---

## 0.1 开工前 30 秒：环境自检

本课程统一使用 conda 环境 **`vllm_learn`**。先确认当前 Python 和 pytest：

```bash
conda activate vllm_learn
cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py
```

如果 `doctor.py` 报 `MISS pytest: No module named 'pytest'`，在你的终端里装：

```bash
conda activate vllm_learn
python -m pip install pytest
```

> `vllm_learn` 里默认没有 pytest；base 环境的 pytest 不能用，因为 base 没有本课程要用的
> torch/transformers 组合。装完 pytest 后重新跑一次 `doctor.py`，直到它说 `ready`。

然后跑一次**初始测试**，确认失败是“TODO 未实现”，不是环境问题：

```bash
cd /home/dictator/ai_infra/nanovllm && python -m pytest tests -q
# 期望：31 failed, 3 passed
```

两个常见误报：

- `cd: too many arguments`：你把 `cd` 和 `pytest` 粘成了一行；用 `&&`，或分两行执行。
- `tests: No such file or directory`：先 `pwd; ls`，确认你在 `nanovllm/` 根目录；不要从
  `mini-v1/` 或工作区根目录直接跑。

---

## 0.2 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm && python -m pytest tests -q
# 34 passed
```

以及四个能脱口而出的答案：

1. `Request` 的哪个字段是 V1 调度的“自变量”？
2. `SchedulerOutput` 为什么分 `scheduled_new_reqs` 和 `scheduled_cached_reqs`？
3. 块哈希为什么必须带 parent？为什么只哈希满块？
4. `SchedulerOutput.validate()` 守的是哪几条不变量？

---

## 1. 先看全景（10 分钟）

在读任何代码之前，把 `docs/00_architecture.md` 的 §1–§4 读一遍。你需要记住这张图：

```text
LLMEngine ──add_request──► EngineCore ──schedule──► SchedulerOutput
                                  ▲                        │
                                  │                        ▼
                           ModelRunnerOutput ◄──── ModelRunner / GPU
                                  │
                                  ▼
                          EngineCoreOutput ──► OutputProcessor ──► RequestOutput
```

本模块只实现最左边的 `Request`、中间的 `SchedulerOutput`、以及前缀缓存要用的块哈希。
`ModelRunnerOutput` / `EngineCoreOutput` 先建好空壳，M06 再用。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`
> 下表的行号已验证；如果漂了，用 `grep -n "<符号>"` 重定位。

### 2.1 必读（按顺序）

| # | 位置 | 符号 | 看什么 |
|---|---|---|---|
| 1 | `vllm/v1/request.py:60` | `class Request` | 字段：`_all_token_ids` / `_output_token_ids` / `num_computed_tokens` / `block_hashes` |
| 2 | `vllm/v1/request.py:272` | `append_output_token_ids()` | 为什么同时改两个 list，然后立刻 `update_block_hashes()` |
| 3 | `vllm/v1/request.py:285` | `update_block_hashes()` | 增量逻辑全部委托给 `_block_hasher` |
| 4 | `vllm/v1/request.py:299` | `num_tokens_with_spec` | 一行 property，但它是整个 V1 调度的目标值 |
| 5 | `vllm/v1/request.py:370` | `RequestStatus` | `is_finished = status > PREEMPTED`，用 IntEnum 排序 |
| 6 | `vllm/v1/core/sched/output.py:41` | `NewRequestData` | 新请求发全量：prompt + block_ids |
| 7 | `vllm/v1/core/sched/output.py:138` | `CachedRequestData` | 老请求发 diff：new_block_ids + 计数 |
| 8 | `vllm/v1/core/sched/output.py:232` | `SchedulerOutput` | 字段注释比代码重要；注意 `num_scheduled_tokens` 与 `total` |
| 9 | `vllm/v1/outputs.py:267` | `ModelRunnerOutput` | 只回增量；list 而不是 tensor |
| 10 | `vllm/v1/engine/__init__.py:199` | `EngineCoreOutput` | 回前端的增量 + `finish_reason` |
| 11 | `vllm/v1/core/kv_cache_utils.py:118` | `NONE_HASH` | 根哈希为什么不是空 bytes |
| 12 | `vllm/v1/core/kv_cache_utils.py:695` | `hash_block_tokens()` | `hash_fn((parent, tuple(tokens), extra_keys))` |
| 13 | `vllm/v1/core/kv_cache_utils.py:941` | `get_request_block_hasher()` | 从 `len(block_hashes)` 开始，只返回新增满块 |

### 2.2 预览（现在只看，不实现）

| 位置 | 符号 | 为什么现在看 |
|---|---|---|
| `vllm/v1/engine/core.py:657` | `EngineCore.step()` | 10 行主循环：schedule → execute → sample → update |
| `vllm/v1/engine/llm_engine.py:319` | `LLMEngine.step()` | 前端如何消费 `EngineCoreOutputs` |
| `vllm/v1/core/sched/scheduler.py:591` | `Scheduler.schedule()` | 只看顶部注释和两个循环的骨架，M05 再细读 |
| `vllm/v1/core/sched/utils.py:98` | `check_stop()` | 停止条件：eos / stop_token_ids / max_tokens / max_model_len |
| `vllm/v1/core/sched/request_queue.py:75` | `FCFSRequestQueue` | 就是一个 deque，M05 直接用 |

### 2.3 明确跳过

`Request` 里的多模态、LoRA、structured output、streaming session、KV transfer 字段；
`SchedulerOutput` 里的 encoder / connector / spec / async 字段；
`RequestStatus` 里的 `WAITING_FOR_*` 状态。**第一遍全部不看。**

---

## 3. 核心概念

### 3.1 `RequestStatus` 用 IntEnum 排序

```python
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
```

`is_finished(status) = status > PREEMPTED`：所有 finish 状态都排在 `PREEMPTED` 后面。
好处是判断完成只需要一次整数比较，不用 `in {...}`。代价是**枚举顺序不能乱**。
`Request.get_finished_reason()` 再把 status 映射成 `FinishReason`（stop/length/abort/error），
M05/M06 输出时用。

### 3.2 三个计数

```text
num_tokens           = len(all_token_ids)          # prompt + 已生成
num_output_tokens    = len(output_token_ids)       # 只算生成
num_tokens_with_spec = num_tokens + len(spec_token_ids)
```

`spec_token_ids` 是投机解码的草稿；本课程不实现，但字段保留，保证公式和 V1 同构。
**注意 `num_tokens` 不包含 spec**；`num_tokens_with_spec` 才是调度目标。

### 3.3 块哈希：链式、只哈希满块、增量

```text
block[0] = H(NONE_HASH,        tokens[0:B])
block[1] = H(block[0],         tokens[B:2B])
block[2] = H(block[1],         tokens[2B:3B])
...
```

三个必须满足的性质：

1. **前缀相同 ⇒ 哈希相同**：这是前缀缓存能复用的前提。
2. **前缀改一个 token ⇒ 后面所有块哈希全变**：这是“只共享完全相同前缀”的保证。
3. **只哈希满块**：尾块还会长出新 token，哈希会变；把它放进缓存会“部分命中”一块只算了一半的 KV。

为什么 parent 不能省？考虑 `[A][B]` 和 `[C][B]`：第二块内容都是 B，但前缀不同。
如果哈希只算 `(tokens)`，它们会命中同一块；带上 parent 后，`H(A,B) != H(C,B)`，正确。

`NONE_HASH` 是 32 字节的**非空**常量，不是 `b""`。这样 `hash_fn((NONE_HASH, ...))` 永远有
非空 parent；也避免了“空 bytes 被当成 falsy”的坑。

**增量**：`Request.append_output_token_ids()` 每来一个 token 都会调 `update_block_hashes()`。
hasher 必须从 `len(request.block_hashes)` 开始算，只返回新块；否则会重复追加，长度爆炸。

### 3.4 `SchedulerOutput`：新请求全量，老请求 diff

vLLM 的 worker 会缓存每个请求的 prompt 和 block table。所以：

- 第一次调度：`NewRequestData(req_id, prompt_token_ids, block_ids, num_computed_tokens)`
- 之后调度：`CachedRequestData(req_ids, new_block_ids, num_computed_tokens, ...)`
- 所有被调度请求：`num_scheduled_tokens[req_id] = n`
- 本步完成：`finished_req_ids`

这样通信量从 `O(D × steps)` 降到 `O(D + steps)`。**这就是为什么 `SchedulerOutput` 要分两个 list。**

### 3.5 四条不变量

`validate()` 要守住（至少）：

1. `total_num_scheduled_tokens == sum(num_scheduled_tokens.values())`
2. `total_num_scheduled_tokens <= max_num_batched_tokens`
3. 每个 `num_scheduled_tokens[req] > 0`
4. `finished_req_ids` 不能出现在 `num_scheduled_tokens` 里

再加三条工程上的：

5. `scheduled_new_reqs` 的 id 必须在 `num_scheduled_tokens` 里；
6. new / cached 的 id 不能重复；
7. `CachedRequestData` 的四个 list 等长，且 id 都在 `num_scheduled_tokens` 里。

用 `assert` 而不是 `ValueError`：和 vLLM 一致，契约错误直接炸，不要静默。

---

## 4. 从 vLLM 到 nanovllm：保留 / 删除

| vLLM | nanovllm | 处理 |
|---|---|---|
| `Request` 的 `mm_features` / `lora_request` / `prompt_embeds` | 删 | 只保留 token ids |
| `RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 等 | 删 | 只保留 7 个状态 |
| `SchedulerOutput` 的 encoder / connector / spec 字段 | 删 | 保留 `scheduled_spec_decode_tokens` 空壳 |
| `NewRequestData.block_ids: tuple[list[int], ...]` | `list[int]` | 单 KV group |
| `CachedRequestData.new_token_ids`（PP 用） | 删 | 单卡不需要 |
| `NONE_HASH` 的随机种子 / 加密哈希 | 简化 | 固定 `sha256("nanovllm-none-hash")` |
| `hash_block_tokens` 的 `extra_keys` | 保留参数 | 传 `None`，为 LoRA/多模态留位 |

---

## 5. 实现任务

> 仓库里已经有数据类的骨架和测试。你的任务是填 `TODO(M00-x)`，不要重写整个文件。
> 每完成一个 TODO，跑一次对应的测试；不要一次改 6 个地方。

### T0 · 跑一遍失败测试（5 分钟）

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests -q
```

你会看到一批 `NotImplementedError`。记下失败数量。**不要现在看 `solutions/`。**

### T1 · `hash_block_tokens()`（20 分钟）

文件：`nanovllm/utils/hashing.py`

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    # 1) parent 为 None 时换成 NONE_HASH
    # 2) 把 (parent, tuple(tokens), extra_keys) 交给 hash_function
    # 3) 保证返回值是 bytes
```

验收：

```bash
python -m pytest tests/test_m00_hashing.py -q -k "hash and not hasher"
```

要点：

- `parent_block_hash` 可能是 `b""` 吗？不会；`NONE_HASH` 是非空 bytes。
- `curr_block_token_ids` 可能是 tuple 或 list；统一 `tuple()` 再哈希，否则 `list` 不可哈希。
- 返回类型必须是 `bytes`；`hash_function` 可能返回 `bytearray`，统一 `bytes(...)`。

### T2 · `get_request_block_hasher()`（30 分钟）

文件：`nanovllm/utils/hashing.py`

```python
def get_request_block_hasher(block_size, hash_function):
    def hasher(request):
        # num_full_blocks = request.num_tokens // block_size
        # 从 len(request.block_hashes) 开始
        # parent 从 request.block_hashes[-1] 开始（没有就是 None）
        # 只返回新块
        ...
    return hasher
```

验收：

```bash
python -m pytest tests/test_m00_hashing.py -q
```

三个坑：

1. **只哈希满块**：`num_full_blocks = num_tokens // block_size`，不是 `ceil`。
2. **增量**：起点是 `len(request.block_hashes)`；重复调用必须幂等。
3. **parent**：第一块用 `None`（内部换成 `NONE_HASH`）；后续块用上一块的哈希。

### T3 · `Request.num_tokens_with_spec`（5 分钟）

文件：`nanovllm/request.py`

一行：

```python
return len(self._all_token_ids) + len(self.spec_token_ids)
```

验收：

```bash
python -m pytest tests/test_m00_request.py -q -k "spec"
```

### T4 · `Request.append_output_token_ids()`（15 分钟）

文件：`nanovllm/request.py`

要求：

- 支持 `int` 和 `list[int]`；
- 同时更新 `_output_token_ids` 和 `_all_token_ids`；
- 最后调用 `self.update_block_hashes()`（新 token 可能凑满一个新块）。

验收：

```bash
python -m pytest tests/test_m00_request.py -q -k "append"
```

### T5 · `Request.update_block_hashes()`（10 分钟）

文件：`nanovllm/request.py`

```python
if self._block_hasher is None:
    return
self.block_hashes.extend(self._block_hasher(self))
```

验收：

```bash
python -m pytest tests/test_m00_request.py -q -k "hash"
```

### T6 · `SchedulerOutput.validate()`（30 分钟）

文件：`nanovllm/sched/output.py`

按 §3.5 的 7 条不变量写 `assert`。每条 assert 带一句人类可读的 message。

验收：

```bash
python -m pytest tests/test_m00_scheduler_output.py -q
```

### T7 · 纸面推演（15 分钟，不写代码）

拿一张纸，画两个请求：

```text
A: prompt = [1,2,3,4,5,6], max_tokens=2
B: prompt = [7,8,9],       max_tokens=2
max_num_batched_tokens = 4, block_size = 4
```

手工填出前 3 步的 `SchedulerOutput`：

- `num_scheduled_tokens`
- `total_num_scheduled_tokens`
- 哪些是 `new`，哪些是 `cached`
- `num_computed_tokens` 每步怎么变

然后把这张纸贴到笔记里。M05 会用到。

---

## 6. 常见坑

**坑 1：把尾块也哈希了。**
尾块会继续长；一旦哈希进缓存，下一个请求拿同样前缀来查，会命中一块“只算了一半”的 KV，
输出直接错。只哈希 `num_tokens // block_size` 个满块。

**坑 2：hasher 每次从 0 开始。**
`append_output_token_ids` 每来一个 token 都调一次；如果每次重算所有块，`block_hashes`
会重复追加，`len(block_hashes)` 和真实块数对不上，M07 的前缀缓存全乱。

**坑 3：parent 用 `b""` 而不是 `NONE_HASH`。**
`b""` 是 falsy，某些分支会把它当成“没有 parent”；而且它和合法的哈希值无法区分。
统一用 32 字节常量。

**坑 4：`num_tokens_with_spec` 把 spec 算进了 `num_tokens`。**
`num_tokens` 是真实 token；spec 是“可能被拒绝的草稿”。混在一起会让停止条件、
prefix cache、logprobs 全错。

**坑 5：`validate()` 只检查 total，不检查 new/cached 的 id。**
调度器组装 `SchedulerOutput` 时最容易犯的错，就是 new 里有、`num_scheduled_tokens`
里没有。worker 会找不到请求，直接 KeyError。

---

## 7. 自测题（做完后不看代码回答）

1. 为什么 `RequestStatus` 用 `IntEnum`，而不是普通 `Enum` 或字符串？
2. `num_computed_tokens` 和 `num_tokens_with_spec` 分别是谁维护的？在什么时机更新？
3. 块哈希为什么要带 parent？用 `[A][B]` vs `[C][B]` 举例。
4. 为什么只哈希满块？尾块什么时候才会被哈希？
5. `SchedulerOutput` 为什么分 `new` 和 `cached`？如果不分，通信量会怎样？
6. `validate()` 的 7 条不变量里，哪一条最容易在 chunked prefill 时被违反？
7. `NONE_HASH` 为什么不能是 `b""`？

---

## 8. 完成定义（DoD）

- [ ] `python -m pytest tests -q` 全绿；
- [ ] 能不看代码画出 `Request → SchedulerOutput → ModelRunnerOutput → EngineCoreOutput` 的流向；
- [ ] 能解释 `num_computed_tokens` 追赶模型，并用 chunked prefill / prefix caching / 抢占各举一例；
- [ ] 能解释链式哈希的三个性质；
- [ ] 能背出 `SchedulerOutput.validate()` 的 7 条不变量；
- [ ] `PROGRESS.md` 已更新；
- [ ] commit：`M00: contracts (request / scheduler output / block hash)`。

---

## 9. 交接：给 M01 的 agent

M01 会用到：

- `nanovllm.request.Request` / `SamplingParams`（已实现）；
- `nanovllm.utils.hashing`（M07 才用，但先保证测试绿）；
- `nanovllm.outputs` 里的数据结构（空壳，M06 再填）。

M01 不需要 `SchedulerOutput`，但需要知道 `Request` 的 `prompt_token_ids` 和
`SamplingParams` 的字段。请在 `PROGRESS.md` 里写清楚：

```text
M00 完成：
- Request / RequestStatus / SamplingParams
- hash_block_tokens / get_request_block_hasher
- SchedulerOutput / NewRequestData / CachedRequestData / validate
已知偏差：
- 没有 spec_token_ids 的实际调度（只保留字段）
- 没有 priority / streaming session
M01 入口：
- 读 docs/M01_model.md
- 模型：Qwen3-0.6B
```

---

## 10. 教练流程（给 agent 用）

1. **5 分钟**：让用户复述 §3.2 的三个计数；答错就回到 §3.2。
2. **20 分钟**：带用户读 §2.1 的 1–5 条，只读 `Request` 相关，不读 `Scheduler`。
3. **30 分钟**：用户做 T1/T2；每写完一个函数跑一次测试；不要直接给实现。
4. **20 分钟**：读 `sched/output.py` 的 3 个类；让用户说出“为什么要分 new/cached”。
5. **30 分钟**：用户做 T3–T6；卡住时给“哪一行错了”的提示，不给完整代码。
6. **15 分钟**：做 T7 纸面推演；让用户把 trace 讲一遍。
7. **10 分钟**：自测题 + 更新 `PROGRESS.md` + commit。

**禁止**：一上来贴 `solutions/m00/`；一次让用户改 3 个文件；跳过测试直接进 M01。
