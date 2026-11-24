# 文档作者 brief D：M05 Scheduler + M06 Engine 端到端

## 通用要求

工作区：`/home/dictator/ai_infra/nanovllm`

先读：

1. `docs/00_architecture.md`（§1–§6，尤其 §3 数据契约、§6.3/§6.6）
2. `docs/00_facts.md`（行号、环境、Qwen3、写作约定）
3. `docs/M00_contracts.md`（风格）
4. `docs/_template.md`
5. `ROADMAP.md`
6. `docs/_briefs/B_kv.md` 的 M03 部分（KVCacheManager 接口）
7. `docs/_briefs/A_model.md` 的 M02 部分（ModelRunner 前置）

只写：

- `docs/M05_scheduler.md`
- `docs/M06_engine.md`

不要改代码/其他文档。每个文件 400–550 行。中文。这两个模块是整门课最核心的，
文档要最详细、最可执行。

---

## M05：Scheduler：token 预算、连续批处理、chunked prefill、抢占

### 目标

实现一个 FCFS、token-budget 驱动的 Scheduler：

- `waiting` / `running` 两条队列；
- `schedule()` 产出 `SchedulerOutput`；
- `update_from_output()` 消费 `ModelRunnerOutput`；
- chunked prefill：长 prompt 分多步；
- 抢占：KV 不够时牺牲最后一个 running，释放块，重置 `num_computed_tokens=0`；
- 停止条件：eos / stop_token_ids / max_tokens / max_model_len；
- CPU 上用 fake model runner 做确定性 trace 测试，不碰 GPU。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/core/sched/scheduler.py:84` | `Scheduler` | 字段：`waiting` / `running` / `requests` / `max_num_scheduled_tokens` |
| `vllm/v1/core/sched/scheduler.py:591` | `schedule()` | **只读顶部注释 + 两个循环的主路径** |
| `:591-660` | schedule 开头 | token_budget / input_budget / 各种列表 |
| `:659-857` | running 循环 | `num_new_tokens`、`allocate_slots`、失败抢占、扣预算 |
| `:870-1400` | waiting 循环 | `get_computed_blocks`、`num_new_tokens`、`allocate_slots`、进 running |
| `:1571-1615` | `_preempt_request()` | free blocks、status=PREEMPTED、computed=0、prepend waiting |
| `:1616-1648` | `_update_after_schedule()` | 调度后 `num_computed_tokens += n`；finished 清空 |
| `:2026-2300` | `update_from_output()` | 逐请求取 sampled tokens、append、check_stop、free、emit |
| `:2493-2511` | `_update_request_with_output()` | 追加 token + `check_stop` 的顺序 |
| `:2616-2648` | `add_request()` | 新请求进 waiting |
| `vllm/v1/core/sched/utils.py:98` | `check_stop()` | eos / stop_token_ids / max_tokens / max_model_len |
| `vllm/v1/core/sched/request_queue.py:75` | `FCFSRequestQueue` | deque |
| `vllm/v1/core/sched/output.py:232` | `SchedulerOutput` | 字段 |
| `vllm/v1/core/kv_cache_manager.py:371` | `allocate_slots()` | 只看签名和返回值 |
| `vllm/v1/request.py:370` | `RequestStatus` | 状态机 |

**明确跳过**：priority、encoder、KV connector、DP/PP、spec decode、structured output、
streaming input、Mamba/sliding window、async scheduling。主路径大约 200 行，其余全是分支。

### 必须讲清的概念

1. **没有 prefill/decode 阶段**：只有 `num_computed_tokens` 追 `num_tokens_with_spec`。
   文档里用 M00 的 5 个例子再讲一遍，并给一个 3 请求的 trace。
2. **token_budget**：`max_num_batched_tokens`；每个 running 请求最多吃
   `num_tokens_with_spec - num_computed_tokens`，但不能超过剩余预算。
3. **running 优先**：先保证已运行的请求；预算不够时抢占最后一个 running（FCFS 下是
   最新进来的），而不是拒绝新请求。
4. **waiting 准入**：只有 `not preempted_this_step` 才调度 waiting（vLLM 的行为）；
   新请求先查前缀缓存，再分配块；分配失败就停，不抢占（waiting 还没有块）。
5. **chunked prefill**：`num_new_tokens = min(remaining, budget)`；没追平就不采样。
6. **new vs cached**：第一次调度的请求进 `scheduled_new_reqs`；之后进
   `scheduled_cached_reqs`；抢占后重新调度算 `resumed_req_ids`。
7. **`_update_after_schedule` 的时机**：先组装 `SchedulerOutput`（使用旧的
   `num_computed_tokens`，worker 需要它来定位 input），再统一 `+= n`。顺序反了，
   worker 会从错误的位置取 token。
8. **`update_from_output` 的过滤**：chunked prefill 的中间步骤，runner 可能采样了
   （因为 `logits_indices = query_start_loc[1:] - 1`）但请求还没追平；Scheduler 必须
   丢弃这些 token。vLLM 用 `discard_request_mask`；nano 可以简单地判断
   `request.num_computed_tokens >= request.num_tokens` 才 append。
9. **抢占是 recompute**：不 swap KV；`num_computed_tokens=0`，下次重新 prefill。
   prefix cache 可能减少重算。
10. **停止顺序**：append token → check_stop → 如果停，free block、从 running 移除、
    emit `finish_reason`。不要先 free 再 append。

### 必须实现的接口

按 `docs/00_architecture.md` §6.3：

```python
class Scheduler:
    def __init__(self, config: SchedulerConfig, kv_cache_manager: KVCacheManager,
                 max_model_len: int): ...
    def add_request(self, request: Request) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_output: ModelRunnerOutput) -> EngineCoreOutputs: ...
    def has_unfinished_requests(self) -> bool: ...
    def finish_requests(self, request_ids, status) -> None: ...
```

内部字段：

```python
self.waiting: deque[Request]
self.running: list[Request]
self.requests: dict[str, Request]
self.finished_req_ids: set[str]   # 上一步完成，待通知 worker
```

### `schedule()` 的简化算法（文档里逐步写）

```text
token_budget = max_num_batched_tokens
num_scheduled = {}
new_reqs, cached_reqs, preempted = [], [], set()
i = 0
while i < len(running) and token_budget > 0:
    req = running[i]
    num_new = min(req.num_tokens_with_spec - req.num_computed_tokens, token_budget)
    if num_new <= 0: i += 1; continue
    blocks = kv.allocate_slots(req, num_new)
    if blocks is None:
        victim = running[-1]
        if victim is req: break
        running.pop()
        kv.free(victim)
        victim.reset_for_preemption()
        waiting.appendleft(victim)
        preempted.add(victim.request_id)
        continue        # 重试当前 req
    num_scheduled[req.id] = num_new
    token_budget -= num_new
    i += 1

if not preempted:
    while waiting and token_budget > 0 and len(running) < max_num_seqs:
        req = waiting[0]
        hit_blocks, num_hit = kv.get_computed_blocks(req)
        num_new = min(req.num_tokens - num_hit, token_budget)
        if num_new <= 0: break
        blocks = kv.allocate_slots(req, num_new, num_hit, hit_blocks)
        if blocks is None: break
        waiting.popleft()
        req.num_computed_tokens = num_hit
        req.status = RUNNING
        running.append(req)
        num_scheduled[req.id] = num_new
        token_budget -= num_new

# 组装 SchedulerOutput
# new: 第一次调度 / resumed；cached: 其余
# 记录 preempted / finished
self._update_after_schedule(out)   # num_computed_tokens += n
return out
```

注意：上面的 `if not preempted` 是 vLLM 的行为（有抢占就本步不接新请求）；nano 可以保留，
并在文档里解释原因（避免抢占后立刻又接纳，抖动）。

### `update_from_output()` 的简化算法

```text
outputs = []
for req_id, n in num_scheduled.items():
    req = requests.get(req_id)
    if req is None or req.is_finished(): continue
    idx = model_output.req_id_to_index[req_id]
    new_ids = model_output.sampled_token_ids[idx]
    if req.num_computed_tokens < req.num_tokens:   # 还在 prefill chunk
        continue
    for tid in new_ids:
        req.append_output_token_ids(tid)
        if check_stop(req, max_model_len):
            req.status = FINISHED_...
            break
    if req.is_finished():
        kv.free(req)
        running.remove(req)
        finished_req_ids.add(req_id)
    if new_ids:
        outputs.append(EngineCoreOutput(req_id, new_ids, finish_reason))
return EngineCoreOutputs(outputs)
```

注意 `num_computed_tokens` 在 `_update_after_schedule` 里已经加过；所以这里判断是否追平
是 `computed >= num_tokens`（append 之前）。

### 测试与对拍

写一个 `FakeModelRunner`：

```python
def execute_model(scheduler_output):
    # 不真的跑模型；按固定规则返回 token（比如 token = hash(req_id) + step）
    # 但必须遵守：chunked prefill 未追平的请求返回 []
```

测试：

1. **FCFS 顺序**：3 个请求，预算足够，第一个先完成。
2. **token budget**：`max_num_batched_tokens=4`，两个 prompt=6，看 trace。
3. **chunked prefill**：prompt=10，budget=4，trace 应为 4/4/2/1...
4. **抢占**：KV 只够 2 个块，3 个请求；检查被抢占的请求 `computed=0`、回到 waiting、
   `num_preemptions=1`、块被释放。
5. **停止条件**：eos、stop_token_ids、max_tokens、max_model_len 各一个 case。
6. **不变量**：每步 `SchedulerOutput.validate(budget)`；`num_scheduled` 的 key 唯一；
   `total == sum`。
7. **无死循环**：所有请求最终 finish；`has_unfinished_requests()` 变 False。
8. **trace 快照**：把每步 `(running ids, waiting ids, scheduled, computed)` 打成表，
   与预期表对比（golden trace）。

### 常见坑

- `num_computed_tokens` 在组装 output 前就加了，worker 取错 token；
- 抢占后没有重试当前请求，或者把当前请求也抢占导致死循环；
- waiting 调度时用了 `request.num_tokens` 而不是 `num_tokens - num_hit`；
- chunked prefill 的中间步 append 了采样 token；
- `finished_req_ids` 没有在下一步传给 worker；
- 完成请求没有从 `running`/`requests` 移除，导致重复调度；
- 预算扣了两次（running 和 waiting 各一次）；
- `max_num_seqs` 没检查；
- prefix hit 的块没有 `touch`，被驱逐。

### 自测题

1. 为什么 V1 没有 prefill/decode 阶段？`num_computed_tokens` 追的是什么？
2. running 和 waiting 的调度顺序为什么不能反？
3. 抢占时为什么把 `num_computed_tokens` 归零，而不是保留？
4. 为什么 `_update_after_schedule` 要在组装 `SchedulerOutput` 之后？
5. chunked prefill 中间步为什么不能 append 采样 token？
6. `finished_req_ids` 的作用是什么？

### 教练流程

概念 25min → 读 running 循环 30min → 读 waiting 循环 30min → 写 `schedule()` 90min →
写 `update_from_output()` 40min → fake runner + trace 测试 60min → 复盘 15min。
先让用户画 trace，再写代码；不要一上来读 900 行。

---

## M06：ModelRunner + InputBatch + EngineCore 端到端

### 目标

把 M01–M05 接起来：

- `InputBatch` / `CachedRequestState` 维护 worker 侧的请求状态；
- `ModelRunner` 实现 `_update_states` / `_prepare_inputs` / `execute_model` /
  `sample_tokens`；
- `GPUWorker` 加载模型、测显存、分配 KV cache；
- `UniProcExecutor` 把 Scheduler 和 ModelRunner 接起来；
- `EngineCore.step()` 跑通 schedule → execute → sample → update；
- `LLM.generate()` 支持多 prompt 连续批处理。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/worker/gpu_model_runner.py:467` | `GPUModelRunner` | `requests` / `input_batch` / `kv_caches` 字段 |
| `:1172-1320` | `_update_states()` | 删 finished、加 new、更新 cached、删 unscheduled |
| `:1544-1600` | `_update_states_after_model_execute()` | 把 sampled token 追加到 worker 侧状态 |
| `:1920-2250` | `_prepare_inputs()` | req_indices、positions、token_indices、query_start_loc、seq_lens、slot mapping、logits_indices |
| `:3940-4100` | `execute_model()` | `_update_states` → `_prepare_inputs` → forward；返回 None 表示等 sample |
| `:4304-4400` | `sample_tokens()` | 取暂存 logits、Sampler、`_update_states_after_model_execute`、返回 ModelRunnerOutput |
| `vllm/v1/worker/gpu_input_batch.py:35` | `CachedRequestState` | worker 侧请求状态 |
| `:90-315` | `InputBatch.__init__` | token_ids_cpu / num_computed / block_table / sampling 数组 |
| `:345-495` | `add_request()` | 初始化行 |
| `:523-580` | `remove_request()` | 删除行、condense |
| `:698-830` | `condense()` | 把删除后的行压缩 |
| `:851-955` | `_make_sampling_metadata()` | 采样数组 |
| `vllm/v1/worker/block_table.py:49` | `BlockTable` | block table / slot mapping |
| `vllm/v1/executor/abstract.py:48` | `Executor` | 接口 |
| `vllm/v1/executor/uniproc_executor.py:34` | `UniProcExecutor` | 单进程直接调 worker |
| `vllm/v1/engine/core.py:657` | `EngineCore.step()` | schedule → execute → sample → update |
| `vllm/v1/engine/llm_engine.py:229/319` | `add_request` / `step` | 前端 |
| `vllm/v1/worker/gpu_worker.py:587` | `load_model()` | 加载模型 |
| `:634-800` | `determine_available_memory()` | profiling 简化 |
| `:853-900` | `initialize_from_config()` | 初始化 KV cache |
| `vllm/v1/engine/output_processor.py:504/681` | `OutputProcessor` | detokenize/输出；M06 只做最小版 |
| `vllm/v1/outputs.py:267` | `ModelRunnerOutput` | 返回格式 |

**明确跳过**：CUDA graph、torch.compile、LoRA、多模态、PP、量化、spec decode、
async scheduling、KV connector、prefix caching（M07 再开）。

### 必须讲清的概念

1. **persistent batch**：worker 侧保留请求状态和 batch 行；每步只做增量更新，
   不要重建整个 batch。
2. **`_update_states` 的三件事**：删 finished、加 new、更新 cached（block ids /
   num_computed / output tokens）；不在本步调度的请求从 batch 移除，但保留 state。
3. **`_prepare_inputs` 的流水线**：
   ```text
   num_scheduled_tokens [num_reqs]
   -> req_indices = repeat(arange(num_reqs), num_scheduled)
   -> cu_num_tokens = cumsum(num_scheduled)
   -> positions = num_computed[req_indices] + arange_in_each_req
   -> input_ids = token_ids_cpu[req_indices, positions]
   -> query_start_loc = [0, cu_num_tokens...]
   -> seq_lens = num_computed + num_scheduled
   -> slot_mapping = block_table.compute_slot_mapping(...)
   -> logits_indices = query_start_loc[1:] - 1
   ```
4. **为什么 `logits_indices` 是每个请求的最后一个 token**：只有最后一个 query 的 logits
   能预测下一个 token；chunked prefill 中间步的 logits 会被 Scheduler 丢弃。
5. **execute/sample 切分**：`execute_model` 做 forward，把 logits 存到
   `self.execute_model_state`，返回 None；`sample_tokens` 取 logits、采样、更新状态、
   返回 `ModelRunnerOutput`。
6. **worker 侧 vs scheduler 侧的两份 Request**：scheduler 的 `Request` 有 status/
   block_hashes；worker 的 `CachedRequestState` 有 prompt/output tokens/block_ids/
   generator。两边通过 `SchedulerOutput` 同步计数和块。
7. **KV cache 初始化**：`num_gpu_blocks = available_bytes // page_size_bytes`；
   每层一个 `[num_blocks, block_size, num_kv_heads, head_dim]` 的 k/v tensor。
8. **`LLM.generate` 的循环**：add all requests → while has_unfinished: step() →
   收集 RequestOutput → 按输入顺序返回。

### 必须实现的接口

```python
# nanovllm/worker/input_batch.py
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: list[int]
    num_computed_tokens: int
    output_token_ids: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None

class InputBatch:
    def add_request(self, req: CachedRequestState) -> int: ...
    def remove_request(self, req_id: str) -> int | None: ...
    def condense(self) -> None: ...
    @property
    def num_reqs(self) -> int: ...
    @property
    def req_ids(self) -> list[str]: ...

# nanovllm/worker/model_runner.py
class ModelRunner:
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None: ...
    def sample_tokens(self) -> ModelRunnerOutput: ...

# nanovllm/worker/gpu_worker.py
class GPUWorker:
    def init_device(self) -> None: ...
    def load_model(self) -> None: ...
    def determine_available_memory(self) -> int: ...
    def initialize_kv_cache(self, num_blocks: int) -> None: ...
    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None: ...
    def sample_tokens(self) -> ModelRunnerOutput: ...

# nanovllm/executor/abstract.py + uniproc_executor.py
class Executor: ...
class UniProcExecutor(Executor): ...

# nanovllm/engine/core.py
class EngineCore:
    def add_request(self, request: Request) -> None: ...
    def abort_requests(self, request_ids: list[str]) -> None: ...
    def step(self) -> EngineCoreOutputs: ...
    def has_unfinished_requests(self) -> bool: ...

# nanovllm/engine/llm_engine.py
class LLMEngine:
    def add_request(self, request_id, prompt, sampling_params) -> None: ...
    def step(self) -> list[RequestOutput]: ...
    def has_unfinished_requests(self) -> bool: ...

# nanovllm/entrypoints/llm.py
class LLM:
    def generate(self, prompts, sampling_params, use_tqdm=True) -> list[RequestOutput]: ...
```

### 实现任务

T1 InputBatch/CachedRequestState；T2 `_update_states`；T3 `_prepare_inputs`；
T4 `execute_model`/`sample_tokens`；T5 GPUWorker load/profile/KV init；T6 UniProcExecutor；
T7 EngineCore.step；T8 LLMEngine/LLM.generate + 最小 OutputProcessor；T9 端到端测试。

### 测试与对拍

- **顺序 vs 批量**：同一组 greedy 请求，逐个 generate 与 8 个并发 generate 的输出逐 token 一致。
- **不同长度**：prompt 长度 1/16/17/100，混合 batch。
- **chunked prefill**：budget 小于 prompt 长度，输出仍正确。
- **finish**：eos/max_tokens/abort 都正确；finished 请求从 batch 移除。
- **KV 释放**：跑 100 个请求后 `block_pool.num_free_blocks` 回到初始值。
- **吞吐**：8 并发 vs 逐个，报告 tokens/s。
- **M02 对拍**：单请求时，M06 的输出与 M02 的连续 cache 输出一致。

### 常见坑

- `_update_states` 里 `finished_req_ids` 和 `new_reqs` 有重叠（abort 后同名重提）；
- `InputBatch.remove_request` 后没有 `condense`，行号错位；
- `num_computed_tokens_cpu` 和 Scheduler 的 `num_computed_tokens` 不同步；
- `positions` 用了 batch 内偏移；
- `slot_mapping` 用了逻辑块号；
- `logits_indices` 取了 `query_start_loc[1:]` 而不是 `-1`；
- sampled token 在 `sample_tokens` 和 `update_from_output` 里各 append 一次（双写）；
- finished 请求的 block 没 free，显存泄漏；
- `execute_model` 返回了 `ModelRunnerOutput`（而不是 None），`EngineCore` 又调 sample；
- GPUWorker 的 `num_gpu_blocks` 没扣掉模型/activation，OOM。

### 自测题

1. `_prepare_inputs` 里 `positions` 和 `input_ids` 怎么算？各需要什么？
2. `logits_indices` 为什么是每个请求的最后一个 token？
3. `execute_model` 返回 None 是什么意思？`EngineCore` 接下来做什么？
4. worker 侧的 `CachedRequestState` 和 scheduler 侧的 `Request` 有什么区别？
5. 一个请求完成后，哪些地方要清理？漏了会怎样？
6. `num_gpu_blocks` 怎么算？Qwen3-0.6B 16GB 大概多少？

### 教练流程

概念 20min → 读 `_prepare_inputs` 40min → 写 InputBatch 60min → 写 runner 90min →
接 EngineCore 60min → 端到端 debug 60min → benchmark 30min → 复盘 15min。
先做 batch=1、greedy、无 chunked，再做多请求/ chunked；不要一上来开 prefix cache。

---

## 两篇文档的衔接

- M05 的 Scheduler 用 M03 的 `KVCacheManager`；M06 把 M04 的 attention 接进 runner。
- M05 的 fake runner 测试是 M06 的基础；M06 的 `ModelRunnerOutput` 必须和 M05 的
  `update_from_output` 对齐。
- M06 完成后，M07 打开 prefix caching，M08 扩展 sampling/output。
- 不要改 `SchedulerOutput` / `ModelRunnerOutput` 的字段；要改先改
  `docs/00_architecture.md` §6。

## 交付前检查

- [ ] 两个文件 400–550 行；
- [ ] M05 有完整的 `schedule()` / `update_from_output()` 伪代码和 8 类测试；
- [ ] M06 有 `_prepare_inputs` 的 10 行流水线和 execute/sample 切分；
- [ ] 两篇都明确“跳过哪些 vLLM 分支”；
- [ ] 有顺序 vs 批量的对拍方法；
- [ ] 没有建议读完整 3400 行 scheduler / 7200 行 runner。
