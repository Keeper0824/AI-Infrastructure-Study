# M06 · ModelRunner + InputBatch + EngineCore 端到端

> **模块目标**：把 M01–M05 接起来——`Scheduler` 说“算什么”，`ModelRunner` 真去算，
> `EngineCore.step()` 把两者串成一条 5 行的循环，`LLM.generate()` 一次喂 8 个 prompt。
>
> **预计工时**：5–6 小时　**依赖**：M01–M05　**测试**：CPU（InputBatch/_update_states）+
> GPU（端到端）　**产出**：`test_m06_engine.py` 全绿；`LLM.generate([p1..p8])` 吞吐高于
> 逐个跑；同一组 greedy 请求“批量 == 逐个”。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m06_engine.py -q           # CPU 部分
python -m pytest tests/test_m06_engine.py -q -m gpu    # 端到端（batch==single/chunked/finish）
python benchmarks/bench_throughput.py --num-prompts 8 --max-tokens 64
```

以及五个能脱口而出的答案：① `positions`/`input_ids` 怎么算；② `logits_indices` 为什么是
每行最后一个 token；③ `execute_model` 返回 `None` 什么意思；④ 两份 `Request` 的区别；
⑤ `num_gpu_blocks` 怎么算、Qwen3-0.6B 16GB 大概多少。

---

## 1. 为什么需要这个模块（10 分钟）

M05 的 Scheduler 只产出 `SchedulerOutput`——一张“本步算什么”的指令单。真正执行需要
三层：`Executor`（谁跑）→ `Worker`（哪张卡）→ `ModelRunner`（怎么把 N 个请求拼成一个
batch 前向）。没有它们，scheduler 永远空转。

关键是 **persistent batch**：worker 保留请求状态和 batch 行，每步只做增量更新；若每步重建
整个 batch（重拷 prompt、重建 block table、重算 metadata），调度省下的时间会全还回去。

| 层 | 职责 | 冻结接口 |
|---|---|---|
| `Executor` | 把 `execute_model` / `sample_tokens` 转发给 worker | `00_architecture.md` §6.6 |
| `GPUWorker` / `ModelRunner` | 建 device、加载权重、测显存、分配 KV；`_update_states`/`_prepare_inputs`/forward/sample | §4 / §6.6 |
| `EngineCore` | `schedule → execute → sample → update` | §6.6 |
| `LLMEngine` + `LLM` | 分词、收集 `RequestOutput`、按输入顺序返回 | §6.7 |

**本模块是整门课最核心的接线**：前面每个模块都能单独测，这里第一次要求它们同时正确。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`；格式
> `vllm/v1/worker/gpu_model_runner.py:1920 · _prepare_inputs()`。行号本机校验过。
> runner 有 7255 行，**主链路约 300 行**，其余是 CUDA graph / 量化 / 多模态 / PP / spec。

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `gpu_model_runner.py:467` / `:1172` | `GPUModelRunner` / `_update_states` | 三个字段（`:573/:683/:714`）；删 finished（:1184）→ 删 unscheduled（:1227）→ 加 new（:1246）→ 更新 cached（:1331）→ add+condense（:1488-1492） | LoRA/MM/PP、spec/streaming |
| 2 | `:1920` / `:3468` | `_prepare_inputs` / `_bookkeeping_sync` | `req_indices`(1944)/`cu`(1948)/`positions`(1953)/`token_indices`(1967)/`qsl`(2030)/`seq_lens`(2139)/slot(2144)/`logits_indices`(2191)；sampled 写回 state（:3586） | mrope/spec/CUDA graph、logprobs |
| 3 | `:3940` / `:4304` | `execute_model` / `sample_tokens` | 先 `_update_states`(3979) → 空则 `EMPTY`(4004) → 前向 → 暂存(4267) → `None`(4286)；sample 取 logits、清 state(4331)、返回 `ModelRunnerOutput` | connector/grammar、spec drafter |
| 4 | `gpu_input_batch.py:35` / `:90` | `CachedRequestState` / `InputBatch` | 请求级字段（prompt/output/block_ids/computed/generator）；行数组 `token_ids_cpu`/`num_computed_cpu`/`block_table`/采样参数 | mm/lora/pooling、预分配 buffer |
| 5 | `gpu_input_batch.py:345/523/698/851` | `add` / `remove` / `condense` / `_make_sampling_metadata` | 行列增删、空洞用最后一行填；采样参数变 tensor（M08 细读） | LoRA/MM、logits processors |
| 6 | `block_table.py:49/153`；`executor/abstract.py:54`；`uniproc_executor.py:51` | `BlockTable` / `Executor` | CPU/GPU 两份表，`slot = block_id * block_size + offset`（:374）；executor 只是转发 | Triton kernel、Ray/MP/PP |
| 7 | `engine/core.py:115/657`；`llm_engine.py:54/229/319` | `EngineCore.step` / `LLMEngine` | schedule → execute → （None 时）sample → update；前端建 `Request`、消费 `EngineCoreOutputs` | batch queue/DP/async、stats |
| 8 | `output_processor.py:504/681`；`gpu_worker.py:436/587/634/853` | `OutputProcessor` / `GPUWorker` | 输出如何变 `RequestOutput`（nano 全量 decode）；显存 profiling：`available = requested - non_kv`（:740） | 增量 detokenize、graph profiling |
| 9 | `outputs.py:267`；`offline.py:589/610` | `ModelRunnerOutput` / `_run_engine` | `sampled_token_ids` 可为空；`while has_unfinished_requests(): step()`，最后排序返回 | prompt logprobs、tqdm |

**明确跳过**：CUDA graph、`torch.compile`、LoRA、多模态、PP、量化、spec、async scheduling、
KV connector、prefix caching（M07）、`_dummy_run`。**不要通读 7255 行**，每行只读 30–60 行。

---

## 3. 核心概念

### 3.1 persistent batch

worker 的 `InputBatch` 里每个请求占一行，行号**动态**（删除留空洞，`condense()` 用最后
一行填）。完整状态留在 `CachedRequestState`，行里只放当前 batch 要用的东西：

```text
CachedRequestState（req_id 索引，跨 step）: prompt/output tokens、block_ids、computed、generator
InputBatch（行号索引，每步可变）: token_ids[row]、computed[row]、block_table[row]、采样参数
```

### 3.2 `_update_states` 的四件事

```text
1) 删 finished：requests.pop(id)；input_batch.remove_request(id)
2) 删 unscheduled：本步没调度的行（被抢占/预算不够）移出 batch，保留 state
3) 加 new：scheduled_new_reqs → CachedRequestState → requests[id] → 待 add_request
4) 更新 cached：num_computed_tokens、output_token_ids、block_ids
最后：add_request → condense() → 刷新 metadata
```

`resumed_req_ids` 的处理不同：worker 的行已被删，`block_ids` 要**整体替换**而不是追加
（`gpu_model_runner.py:1417-1427`）。顺序也不能反：删完再 `condense`，否则行号错位。

### 3.3 `_prepare_inputs` 的十条流水线

```text
num_scheduled_tokens [num_reqs]            # 来自 scheduler_output，按 batch 行序
-> req_indices   = repeat(arange(num_reqs), num_scheduled)      # 每个 token 属于哪行
-> cu_num_tokens = cumsum(num_scheduled)                        # 行内扁平偏移
-> positions     = num_computed[req_indices] + arange_in_each_req
-> input_ids     = token_ids_cpu[req_indices, positions]      # 绝对位置查表
-> query_start_loc = [0, cu_num_tokens...]                      # [num_reqs+1]
-> seq_lens      = num_computed + num_scheduled                 # 每行“总长”
-> slot_mapping  = block_table.compute_slot_mapping(query_start_loc, positions)
-> logits_indices = query_start_loc[1:] - 1                     # 每行最后一个 query
-> AttentionMetadata(num_reqs, total, max_query_len, ...)
```

**数值例子**（`block_size=4`，两行；A 的 block table `[7,9]`，B 的 `[3]`）：

```text
num_scheduled = [2, 3]   num_computed = [4, 1]   req_indices = [0,0,1,1,1]
positions = [4, 5, 1, 2, 3]   query_start_loc = [0, 2, 5]   seq_lens = [6, 4]
slot_mapping = [36, 37, 13, 14, 15]   # A: pos4,5 -> 块 9 的 offset 0,1
logits_indices = [1, 4]               # -> logits [2, vocab]
```

注意 `positions` 是**绝对位置**（`num_computed` 是序列内偏移，不是 batch 内偏移）；
`slot_mapping` 必须先查 block table（`pos // block_size` 只是**逻辑**块号）。

### 3.4 `logits_indices` 为什么是每行最后一个 token

只有最后一个 query 的 hidden state 见过整行上下文，其 logits 才能预测下一个 token。
chunked prefill 的中间步也算 logits（kernel 形状统一），但 scheduler 会丢掉（M05 §3.8）：

```python
logits_indices = query_start_loc[1:] - 1        # gpu_model_runner.py:2191
discard_request_mask = optimistic_seq_lens < num_tokens   # :2058
```

nano 在 runner 侧也判一次：`num_computed + n < num_tokens` 的行，**采样后丢弃、不写回
state**，`sampled_token_ids[row] = []`。Scheduler 侧再判一次，两边一致才不会错位。

### 3.5 execute / sample 切分

```text
execute_model(scheduler_output) -> None | ModelRunnerOutput
    _update_states → _prepare_inputs → model forward → 存 logits → return None
sample_tokens() -> ModelRunnerOutput
    取 logits → argmax（M08 换成 Sampler）→ 丢弃未追平的行 → 写回 state → 返回
EngineCore.step():  out = executor.execute_model(so)
                    if out is None: out = executor.sample_tokens()
                    scheduler.update_from_output(so, out)
```

返回 `None` 的语义：**前向做完了，logits 暂存，等我调 `sample_tokens`**。返回
`ModelRunnerOutput` 的语义：**本步没有需要采样的 token**（空 batch / 只有 finished），
`EngineCore` 不再调 `sample_tokens`。这两种情况不能混。

### 3.6 两份 Request：scheduler 侧 vs worker 侧

| | scheduler `Request` | worker `CachedRequestState` |
|---|---|---|
| 谁维护 | Scheduler | ModelRunner |
| 关键字段 | `status`、`num_computed_tokens`、`block_hashes` | `prompt_token_ids`、`output_token_ids`、`block_ids`、`generator` |
| 怎么同步 | 写 `SchedulerOutput` | 读 `SchedulerOutput`，写 `ModelRunnerOutput` |
| 谁 append token | `update_from_output` | `sample_tokens`（`_bookkeeping_sync`） |

两份 `output_token_ids` 是**故意的**：worker 算 `num_tokens`/`input_ids`，scheduler 判停。
**两边都要 append，且只 append 一次**，否则会漂移（§7 坑 4）。

### 3.7 KV cache 初始化

```text
page_size_bytes = block_size * num_kv_heads * 2 * head_dim * dtype_size   # 每层每块
num_gpu_blocks  = available_kv_bytes // page_size_bytes
每层 k/v: [num_gpu_blocks, block_size, num_kv_heads, head_dim]           # §6.5 的 layout
```

Qwen3-0.6B（28 层、8 个 KV head、head_dim 128、bf16、block 16）：每层每块
`16*8*2*128*2 = 64 KiB`，全模型每块 1.75 MiB，每 token 112 KiB。16GB 卡按 0.85 利用率、
减 1.2GB 权重和约 2GB 激活，KV 约 10.5GiB → **约 6000 块、9.6 万 token**；真实值以
`determine_available_memory()` 为准。

### 3.8 `LLM.generate` 的循环

```text
1) 分词、建 Request（带 eos_token_id）、engine.add_request
2) while engine.has_unfinished_requests():
       step_outputs = engine.step()            # 每个 finished 的 RequestOutput 收集
3) 按输入顺序返回（vLLM 按 request id 排序，nano 用 client_index 排序）
```

`step()` 一步推进一个 batch；8 个 prompt 在同一个循环里被动态拼批（M05 的 waiting 准入）。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `GPUModelRunner(vllm_config, device)` | `ModelRunner(config, kv_cache_manager, device)` | 删 graph/compile/quant/MM/PP |
| `InputBatch` 预分配 pinned buffer | Python list + dict | 先正确，M10 再预分配 |
| `CachedRequestState.block_ids: tuple[list, ...]` | `list[int]` | 单 group |
| `_update_states_after_model_execute`（hybrid/MTP 用）；`Executor.collective_rpc` | 不需要；直接 `self.worker.execute_model(...)` | 单进程 |
| `EngineCore.step() -> tuple[outputs, bool]` | `-> EngineCoreOutputs` | 简化 |
| `LLMEngine.step()` 从 `get_output()` 取；增量 detokenize；`Sampler` 九步 | 直接调 `engine_core.step()`；全量 decode；`argmax(-1)` | 同步；M08 再替换 |
| KV cache layout `[blocks, kv_heads, block, 2*head]`（Triton 后端） | `[blocks, block, kv_heads, head]` | 按 §6.5 冻结 |

**本模块接口**（§6.6/§6.7 已冻结；`InputBatch`/`GPUWorker`/`CachedRequestState` 是
§6 未列、由本文固定，属于“本模块新增”，M07/M08 依赖）：

```python
@dataclass
class CachedRequestState:
    req_id: str; prompt_token_ids: list[int]; sampling_params: SamplingParams
    block_ids: list[int]; num_computed_tokens: int
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

class GPUWorker:
    def init_device(self) -> None: ...
    def load_model(self) -> None: ...
    def determine_available_memory(self) -> int: ...
    def initialize_kv_cache(self, num_blocks: int) -> None: ...
    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None: ...
    def sample_tokens(self) -> ModelRunnerOutput: ...
```

`ModelRunner` / `Executor` / `EngineCore` / `LLMEngine` / `LLM` 的签名严格用 §6.6、§6.7。

---

## 5. 实现任务

> 顺序即依赖：T1→T2→T3→T4 可以纯 CPU 测；T5 起要 GPU。每个任务先写失败测试。

### T1 · `CachedRequestState` + `InputBatch`（60 分钟）

文件：`nanovllm/worker/input_batch.py`

```python
def add_request(self, req):        # 找空洞或追加一行；写 token_ids/num_computed/block_table
def remove_request(self, req_id):  # 返回旧行号；清空该行的 token/block_table/sampling
def condense(self):                # 用最后一行填最小空洞，更新 req_id_to_index
```

验收：`pytest ... -k input_batch`。覆盖 add→num_reqs、remove 后 `condense` 行号连续、删中间
行后最后一行被搬、重复 add 报错。坑：① 没 `condense` 行号错位；② 搬 token 忘了 `move_row`；
③ `req_ids` 残留 `None`。

### T2 · `_update_states`（60 分钟）

文件：`nanovllm/worker/model_runner.py`，纯 CPU 可测。

```python
def _update_states(self, so):
    # 1) finished：requests.pop + input_batch.remove_request
    # 2) unscheduled = batch 里有、但不在 so.num_scheduled_tokens 里 -> remove
    # 3) new：构造 CachedRequestState（generator 用 sampling_params.seed）
    # 4) cached：num_computed_tokens = so 里的值；block_ids 追加 / resumed 整体替换
    # 5) 新行 add_request；最后 condense()
```

验收：`pytest ... -k update_states`。覆盖正常增删、抢占 resume、finished 与 new 同 id、
未调度请求移出 batch 但 state 保留。坑：① 先 add 再 remove；② resumed 用 append；③ 忘记
`finished_req_ids` 里已不在 batch 的 id。

### T3 · `_prepare_inputs` + `BlockTable`（60 分钟）

文件：`nanovllm/worker/model_runner.py`（`BlockTable` 用 M03 的，不要重写）

```python
def _prepare_inputs(self, so) -> tuple[Tensor, Tensor, AttentionMetadata, Tensor]:
    # §3.3 的十条流水线；block_table 取 M03 的 .block_table[:num_reqs] 转 GPU int32
    # 返回 (input_ids, positions, attn_metadata, logits_indices)
```

验收：`pytest ... -k prepare_inputs`，用 §3.3 的两行例子逐项断言（CPU tensor）。坑：
① `positions` 用行内偏移；② `slot_mapping` 用逻辑块号；③ `logits_indices` 漏 `-1`。

### T4 · `execute_model` / `sample_tokens`（90 分钟）

```python
def execute_model(self, so):
    self._update_states(so)
    if so.total_num_scheduled_tokens == 0: return EMPTY_MODEL_RUNNER_OUTPUT
    input_ids, positions, meta, logits_idx = self._prepare_inputs(so)
    hidden = self.model(input_ids, positions, self.kv_caches, meta)
    self.execute_model_state = (so, self.model.compute_logits(hidden[logits_idx]), logits_idx)
    return None

def sample_tokens(self):
    so, logits, _ = self.execute_model_state; self.execute_model_state = None
    ids = logits.argmax(-1)                    # M08: self.sampler.forward(logits, meta)
    return self._finalize(so, ids)             # 私有：丢弃未追平行 + 写回 state + 组装输出
```

验收：`pytest ... -m gpu -k runner`。覆盖空 batch 返回 `EMPTY_MODEL_RUNNER_OUTPUT`、两次
`execute_model` 之间必须 `sample_tokens`、未追平行返回 `[]`。坑：① 空 batch 不调
`_update_states`；② 返回 output 又让 EngineCore 调 sample；③ 对未追平行也 append。

### T5 · `GPUWorker`（60 分钟）

文件：`nanovllm/worker/gpu_worker.py`

```text
init_device: torch.cuda.set_device；建 ModelRunner
load_model:  M01 的 load_weights
determine_available_memory: mem_get_info -> 用最大 batch 跑一次 dummy forward -> 再 mem_get_info
                           available = total*util - weights - activation_headroom
initialize_kv_cache: 每层 k/v = empty([num_blocks, block_size, kv_heads, head_dim])
execute_model / sample_tokens: 直接转发
```

验收：`pytest tests/test_m06_engine.py -q -m gpu -k "worker"`：打印 `num_gpu_blocks`，
与 §3.7 的估算同量级（5000–7000），且 < 总显存。坑：① 没减激活，OOM；② 每层共享同一个
tensor（`[t]*28` 是同一块内存）；③ `num_blocks` 传成 `num_blocks_per_layer`。

### T6 · `Executor`（30 分钟）

`nanovllm/executor/{abstract,uniproc_executor}.py`：`Executor` 是 ABC，`UniProcExecutor` 持有
`worker`，两个方法各一行转发。验收：`pytest ... -k executor`（fake worker 断言透传）。

### T7 · `EngineCore.step()`（40 分钟）

```python
def step(self):
    if not self.scheduler.has_unfinished_requests(): return EngineCoreOutputs()
    so = self.scheduler.schedule()
    out = self.executor.execute_model(so)
    if out is None: out = self.executor.sample_tokens()
    return self.scheduler.update_from_output(so, out)
```

验收：`pytest tests/test_m06_engine.py -q -k "engine_core"`（fake scheduler + fake executor：
`None` 时调 sample、非 `None` 时不调）。坑：① 没判断 `has_unfinished_requests`，空转；
② `out is None` 写成 `if not out`（空 `ModelRunnerOutput` 是 falsy 吗？dataclass 默认
truthy，但别赌）；③ 忘了把 `so` 传给 `update_from_output`。

### T8 · `LLMEngine` + `OutputProcessor` + `LLM.generate`（60 分钟）

```python
# llm_engine.py
def add_request(self, request_id, prompt, sampling_params): ...   # 分词 + Request + core.add_request
def step(self): return self.output_processor.process_outputs(self.engine_core.step())
def has_unfinished_requests(self): return self.engine_core.has_unfinished_requests()

# processor.py
def add_request(self, request: Request, prompt: str) -> None: ...
    # prompt 是原文（Request 只有 prompt_token_ids）；RequestOutput.prompt 要用
def process_outputs(self, outputs: EngineCoreOutputs) -> list[RequestOutput]: ...
    # 每步对每个 EngineCoreOutput 的 new_token_ids 做 tokenizer.decode(全部 ids)

# outputs.py：RequestOutput(finished=True) 只在该请求 finish 的那步产出
# 这个 add_request 的 prompt 参数是 M08 的冻结接缝，M06 先按它建骨架

# entrypoints/llm.py
def generate(self, prompts, sampling_params, use_tqdm=True):
    # add 全部 -> while has_unfinished: step() 收集 finished -> 按输入顺序返回
```

验收：`pytest tests/test_m06_engine.py -q -m gpu -k "generate"`：单 prompt 输出与 M02
CLI 一致；8 个 prompt 一次 `generate` 全部完成且顺序正确。

### T9 · 端到端测试与 benchmark（60 分钟）

见 §6。验收：`pytest ... -m gpu` 全绿；`bench_throughput.py --num-prompts 8 --max-tokens 64`。

---

## 6. 测试与对拍

### 6.1 顺序 vs 批量（最重要的一条）

```python
batch = llm.generate(prompts, greedy)                 # 8 个不同长度
single = [llm.generate([p], greedy)[0] for p in prompts]
assert [b.outputs[0].token_ids for b in batch] == [s.outputs[0].token_ids for s in single]
```

先比 token，再比 logits。bf16 下不同 batch 形状的 GEMM 可能有极小数值差；如果偶发差 1 个
token，用 `torch.testing.assert_close(logits, atol=2e-2, rtol=1e-2)` 确认是数值而非逻辑。
**不要用 `torch.equal` 比 logits。**

### 6.2 测试清单

| # | 测试 | 参数 | 断言 |
|---|---|---|---|
| 1 | 批量 == 逐个 | 8 个 greedy prompt，`max_tokens=32`；长度 1/16/17/100 混合 | 逐 token 相等（见 §6.1） |
| 2 | chunked prefill | `max_num_batched_tokens=16`，prompt=100 | 与不 chunk 的输出一致（§6.3） |
| 3 | finish | eos / max_tokens / abort 各一例 | `finish_reason` 正确；该行从 `input_batch` 移除 |
| 4 | KV 释放 | 跑 100 个请求 | 结束时 `block_pool.num_free_blocks == 初始值` |
| 5 | 吞吐 + M02 对拍 | 8 并发 vs 逐个；单请求 greedy 同 prompt | tokens/s（§6.4）；与 `ContiguousKVCache` 逐 token 一致 |
| 6 | 空转/清理 | 最后一个请求完成后再走一步 | `finished_req_ids` 被消费；batch 为空 |

### 6.3 chunked prefill 对拍

同一 prompt 用 `budget=16`（强制分块）与 `budget=8192` 各跑一遍 greedy 比 token。分块时
`positions` 必须连续；第二步开始乱码就先查它是不是用了块内偏移。

### 6.4 吞吐测量

```python
t0 = time.perf_counter(); outs = llm.generate(prompts, sp); dt = time.perf_counter() - t0
tokens = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"{tokens/dt:.1f} output tok/s, {len(prompts)} reqs, {dt:.2f}s")
```

同时跑一次“逐个 generate”，报告两者的 tokens/s 和 TTFT。**性能数字必须和 §6.1 的
正确性断言一起出现**，否则不算通过。

### 6.5 失败现象 → 先查

| 现象 | 先查 |
|---|---|
| 批量与逐个从第 2 个 token 起分叉 / 长 prompt 后半段乱码 | `positions` 是否绝对；slot 是否查表；`computed` 两边是否同步 |
| 输出比预期长 | 未追平的行也 append 了 token（§3.4） |
| 第 2 个请求开始重复第一个的 token | block table 行没 `clear_row`/`add_row`，两行指向同一块 |
| 几十步后 OOM / `KeyError: req_id` | finished 没 free、行没删；`ModelRunnerOutput.req_ids` 与 `num_scheduled_tokens` 不一致 |
| `sample_tokens` 报 state 为 None | 上一步 `execute_model` 直接返回了 output，EngineCore 仍调 sample |

---

## 7. 常见坑

1. **`_update_states` 里 finished 与 new 重叠（abort 后同名重提）；`remove_request` 后没
   `condense`**：先删旧再加新（vLLM 当两个请求，`:1191-1195`）；否则行号错位、A 的 KV 写给 B。
2. **`num_computed_tokens_cpu` 与 scheduler 不同步**：worker 只信 `SchedulerOutput` 里的
   值（本步起点），不要自己提前 `+= n`。
3. **`positions` 用了 batch 内偏移 / `slot_mapping` 用了逻辑块号**：`positions = computed[row]
   + cu 内偏移`；slot 必须 `block_table[row, pos//block_size]*block_size + pos%block_size`。
4. **`logits_indices` 漏 `- 1` / sampled token 双写或不写**：前者取到下一请求的首 token；
   后者让 `num_tokens` 每步 +2 或停更。`sample_tokens` 与 `update_from_output` 各写一份。
5. **finished 请求的 block 没 free**：`num_free_blocks` 单调下降，最后 OOM。
6. **`execute_model` 返回了 output**：`EngineCore` 又调 `sample_tokens`，state 为 None。
7. **`num_gpu_blocks` 没扣模型/激活**：初始化或跑几步后 OOM。
8. **`kv_caches` 用 `[tensor] * 28`**：28 层共享同一块显存，第 2 层覆盖第 1 层。
9. **`LLM.generate` 用 `set` 收集输出**：返回顺序不等于输入顺序。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m06_engine.py -q` 与 `-m gpu` 全绿；
- [ ] `LLM.generate` 8 并发输出与逐个 greedy 逐 token 一致；
- [ ] chunked prefill（budget=16、prompt=100）与一次算完一致；
- [ ] eos / max_tokens / abort 三种 finish 正确，且行被移除；
- [ ] 100 个请求跑完后 `num_free_blocks` 回到初始值；
- [ ] `bench_throughput.py` 报出 8 并发 vs 逐个的 tokens/s；
- [ ] `PROGRESS.md` 更新；commit：`M06: engine end-to-end (input batch / runner / core / generate)`。

---

## 9. 交接：给 M07 / M08

**新增文件**

```text
nanovllm/worker/{input_batch,model_runner,gpu_worker}.py
nanovllm/executor/{abstract,uniproc_executor}.py
nanovllm/engine/{core,llm_engine,processor}.py
nanovllm/entrypoints/llm.py
tests/test_m06_engine.py        benchmarks/bench_throughput.py
```

**公开 API**：§4 的 `CachedRequestState` / `InputBatch` / `GPUWorker`；`ModelRunner` /
`Executor` / `EngineCore` / `LLMEngine` / `LLM` 严格按 §6.6/§6.7。

**给 M07（prefix caching）**：`_update_states` 必须能处理 resumed 请求的完整块表；
`_prepare_inputs` 的 `slot_mapping` 已经走 block table，M07 只改 Scheduler/KVManager，
runner 不用动。

**给 M08（采样/输出）**：两处接缝——`sample_tokens` 里的 `logits.argmax(-1)` 换成
`self.sampler.forward(logits, metadata)`；`OutputProcessor.process_outputs` 换成增量
detokenize + `SamplingMetadata`。`ModelRunnerOutput` 的字段不要改。

**已知限制**：单进程、同步、无 CUDA graph、无 prefix cache、只有 greedy、全量 decode、
`n=1`。

---

## 10. 自测题（做完后不看代码回答）

1. `_prepare_inputs` 里 `positions` 和 `input_ids` 分别怎么算？各需要什么张量？
2. `logits_indices` 为什么是每行最后一个 query？中间 chunk 的 logits 去哪了？
3. `execute_model` 返回 `None` 是什么意思？`EngineCore` 接下来做什么？
4. worker 的 `CachedRequestState` 和 scheduler 的 `Request` 有什么区别？谁 append token？
5. 一个请求完成后，哪些地方要清理？漏了会怎样？
6. `num_gpu_blocks` 怎么算？Qwen3-0.6B 16GB 大概多少块、多少 token？

---

## 11. 教练流程（给 agent 用）

| 时间 | 做什么 | 禁止 |
|---|---|---|
| 20min | 讲 §3.1/§3.5；让用户画 execute/sample 的时序 | 讲 CUDA graph |
| 40min | 读 `_prepare_inputs` 的十条流水线 + §3.3 例子 | 通读 7255 行 |
| 60min | T1 InputBatch（纯 CPU 测试） | 上 GPU |
| 90min | T2/T3/T4；先 batch=1、greedy、不 chunk | 一上来开 8 并发 |
| 60min | T5/T6/T7；`num_gpu_blocks` 打出来对账 | 硬编码块数 |
| 60min | T8/T9；先跑单请求，再 8 并发，再 chunked | 跳过“批量==逐个” |
| 30min | benchmark + 复盘 §7 坑 4/7/8 | 只报性能不报正确性 |

顺序原则：**先 batch=1 跑通，再开并发，最后开 chunked**；prefix cache 留给 M07。
