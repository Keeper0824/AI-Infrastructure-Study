# 00 · vLLM V1 全景与 nanovllm 的冻结契约

> 本文件是整门课的总图。**每个模块开工前先读这一篇**；模块文档只讲该模块的细节，
> 不再重复这里的数据流和接口。
>
> 版本锚点：`/home/dictator/ai_infra/target/vllm-0.32.0/`（下称 `@v0.32.0`）。所有 `文件:行号` 都以这棵树为准。
> 行号会漂，符号不会：引用格式统一写成 `vllm/v1/request.py:60 · class Request`，
> 找不到行号时先 `grep -n "class Request"` 重定位，再读附近的代码。

---

## 1. 一句话：V1 到底在做什么

V1 把一次推理拆成**四个可独立替换的部件**：

```
                 ┌─────────────────────────────────────────────────────────┐
                 │   Frontend（前端）                                       │
                 │   LLM / LLMEngine / AsyncLLM                            │
                 │   分词、采样参数校验、detokenize、组装 RequestOutput       │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ EngineCoreRequest（不可变输入）
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │   EngineCore（引擎内核，单进程/单线程忙循环）              │
                 │                                                         │
                 │   Scheduler ──SchedulerOutput──► Executor ──► Worker     │
                 │       ▲                              │                   │
                 │       │                              ▼                   │
                 │       └──── ModelRunnerOutput ◄── GPUModelRunner          │
                 │                                                         │
                 │   KVCacheManager / BlockPool（逻辑块 ↔ 物理显存页）        │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ EngineCoreOutput（增量 token + finish_reason）
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │   Frontend 的 OutputProcessor                           │
                 │   IncrementalDetokenizer → RequestOutput / SSE chunk     │
                 └─────────────────────────────────────────────────────────┘
```

一句话：**Scheduler 只决定“这一步每个请求算几个 token”；ModelRunner 只负责“把这些 token 算出来并采样”；两者之间只通过两个不可变数据结构通信。**

这就是本课程要复刻的骨架。nano 版会把多进程、多模态、量化、TP/PP、MLA/Mamba、LoRA、KV connector 全部删掉，但**保留这条主链路和三层契约**。

---

## 2. 一次 `step()` 的完整时序

以离线 `LLM.generate()` 为例（同步、单进程版）：

```
用户                 LLMEngine                 EngineCore              Scheduler            ModelRunner
 │  add_request()        │                         │                       │                    │
 │──────────────────────►│  InputProcessor          │                       │                    │
 │                       │  Request + OutputState   │                       │                    │
 │                       │────────────────────────►│  add_request()        │                    │
 │                       │                         │──────────────────────►│  waiting.append    │
 │                       │                         │                       │                    │
 │  step()               │                         │                       │                    │
 │──────────────────────►│  engine_core.step()     │                       │                    │
 │                       │────────────────────────►│  scheduler.schedule() │                    │
 │                       │                         │──────────────────────►│                    │
 │                       │                         │  SchedulerOutput      │                    │
 │                       │                         │◄──────────────────────│                    │
 │                       │                         │  execute_model()      │                    │
 │                       │                         │───────────────────────────────────────────►│
 │                       │                         │                       │   forward + KV     │
 │                       │                         │                       │   write + logits   │
 │                       │                         │  sample_tokens()      │                    │
 │                       │                         │───────────────────────────────────────────►│
 │                       │                         │  ModelRunnerOutput    │                    │
 │                       │                         │◄───────────────────────────────────────────│
 │                       │                         │  update_from_output() │                    │
 │                       │                         │──────────────────────►│  append token      │
 │                       │                         │  EngineCoreOutputs    │  check_stop/free   │
 │                       │                         │◄──────────────────────│                    │
 │                       │  process_outputs()      │                       │                    │
 │                       │  detokenize / logprobs  │                       │                    │
 │  RequestOutput        │                         │                       │                    │
 │◄──────────────────────│                         │                       │                    │
```

对应 vLLM 的入口：

| 步骤 | vLLM 位置 | 你要在 nano 里实现的 |
|---|---|---|
| 前端建请求 | `vllm/v1/engine/llm_engine.py:229 · add_request()` | `LLMEngine.add_request()` |
| 引擎单步 | `vllm/v1/engine/core.py:657 · EngineCore.step()` | `EngineCore.step()` |
| 调度 | `vllm/v1/core/sched/scheduler.py:591 · Scheduler.schedule()` | `Scheduler.schedule()` |
| 执行 | `vllm/v1/executor/abstract.py` + `uniproc_executor.py` | `Executor` / `UniProcExecutor` |
| 前向 | `vllm/v1/worker/gpu_model_runner.py:3940 · execute_model()` | `ModelRunner.execute_model()` |
| 采样 | `vllm/v1/worker/gpu_model_runner.py:4304 · sample_tokens()` | `ModelRunner.sample_tokens()` |
| 回写 | `vllm/v1/core/sched/scheduler.py:2026 · update_from_output()` | `Scheduler.update_from_output()` |
| 前端输出 | `vllm/v1/engine/output_processor.py:681 · process_outputs()` | `OutputProcessor.process_outputs()` |

> **V1 的关键细节**：`execute_model()` 和 `sample_tokens()` 是分开的。
> `execute_model()` 只做前向，返回 `None` 并把 logits 暂存在 runner 里；
> `EngineCore` 看到 `None` 才调用 `sample_tokens()`。这不是为了好看，而是为了让
> “GPU 前向”和“CPU 调度/输出处理”可以流水起来（见 `EngineCore.step()` 的
> `future = execute_model(..., non_block=True)`）。nano 版也保留这个切分，
> 但先同步执行；M09/M10 再解释它为什么能异步。

---

## 3. 四个核心数据结构

V1 的通信只靠这四个东西（外加一个 `Request`）。先把它们背下来：

| 结构 | 性质 | 谁写 | 谁读 | 关键字段 |
|---|---|---|---|---|
| `Request` | **可变**，请求的完整生命周期状态 | 前端创建；Scheduler/KVManager 修改 | Scheduler、ModelRunner 的 `CachedRequestState` | `status`、`num_computed_tokens`、`num_tokens_with_spec`、`block_hashes` |
| `SchedulerOutput` | **不可变**，本步“算什么”的指令 | Scheduler | Executor / ModelRunner | `scheduled_new_reqs`、`scheduled_cached_reqs`、`num_scheduled_tokens`、`total_num_scheduled_tokens`、`finished_req_ids` |
| `ModelRunnerOutput` | **不可变**，本步“算出了什么” | ModelRunner | Scheduler | `req_ids`、`req_id_to_index`、`sampled_token_ids` |
| `EngineCoreOutput` | **不可变**，回给前端的增量 | Scheduler | OutputProcessor | `request_id`、`new_token_ids`、`finish_reason` |

### 3.1 `Request`：唯一会“动”的状态

`vllm/v1/request.py:60 · class Request`

调度器眼里的请求只有三个数：

```text
num_tokens            = prompt + 已生成 output
num_tokens_with_spec  = num_tokens + 草稿 token（spec decode 用；nano 先恒为 0）
num_computed_tokens   = 已经真正算进 KV cache 的 token 数
```

`Scheduler.schedule()` 的全部工作，就是让 `num_computed_tokens` **追平** `num_tokens_with_spec`：

```text
                     num_tokens_with_spec
                             │
     已算进 KV 的部分         │  还没算的部分
     ├───────────────────────┼──────────────┤
     ▲                       ▲
     0                       num_computed_tokens
```

- 首次 prefill：`num_computed_tokens` 从 0 追到 `len(prompt)`；
- chunked prefill：一次只追 `chunk_size`，分多步追平；
- decode：每步追 1（生成 1 个 token 后，目标 +1）；
- prefix caching：从“命中的块数 × block_size”开始追，而不是 0；
- 抢占：`num_computed_tokens` 归零，重新追；
- spec decode：目标里多算了草稿 token，接受/拒绝后再修正。

> 这就是 V1 最值钱的一句话：**没有 prefill 阶段，也没有 decode 阶段，只有“追平”。**
> 见 `vllm/v1/core/sched/scheduler.py:591` 顶部注释（原话在 `schedule()` 的 docstring 前）。

### 3.2 `SchedulerOutput`：新请求发全量，老请求发 diff

`vllm/v1/core/sched/output.py:41 · NewRequestData`、`:138 · CachedRequestData`、`:232 · SchedulerOutput`

vLLM 的 worker 进程会缓存每个请求的 prompt、block table、采样参数。所以：

- **第一次被调度**的请求 → `scheduled_new_reqs`，发 `prompt_token_ids` + 全量 `block_ids` + `num_computed_tokens`；
- **已经调度过**的请求 → `scheduled_cached_reqs`，只发 `new_block_ids`、新的 `num_computed_tokens`、`num_output_tokens`；
- 所有被调度的请求 → `num_scheduled_tokens: dict[req_id, int]`；
- 本步结束/被抢占的请求 → `finished_req_ids` / `preempted_req_ids`，通知 worker 清缓存。

这样 D 个请求跑 1000 步，通信量是 `O(D + steps)`，而不是 `O(D × steps)`。

### 3.3 `ModelRunnerOutput`：只回增量

`vllm/v1/outputs.py:267 · class ModelRunnerOutput`

```python
req_ids: list[str]                 # 本步参与 batch 的请求
req_id_to_index: dict[str, int]    # req_id -> 行号
sampled_token_ids: list[list[int]] # 每行本步新生成的 token（chunked prefill 可能为空）
logprobs: LogprobsLists | None
```

注意它是 **list**，不是 GPU tensor。vLLM 注释写得很清楚：`ModelRunnerOutput` 会被序列化送到调度进程，传 tensor 太贵，所以在 worker 里就 D2H 转成 list。

### 3.4 `EngineCoreOutput`：调度器给前端的增量

`vllm/v1/engine/__init__.py:199 · EngineCoreOutput`

```python
request_id: str
new_token_ids: list[int]
finish_reason: FinishReason | None   # stop / length / abort / error
stop_reason: int | str | None
new_logprobs: ...
```

前端拿到它之后才做 detokenize、stop string 检查、组装 `RequestOutput`。

---

## 4. `num_computed_tokens` 的五个例子

这一节建议亲手在纸上推一遍。后面的 Scheduler 模块会直接用到。

### 例 1：普通 prefill + decode

```text
prompt = [10,11,12,13,14], max_tokens = 3, block_size = 4
num_tokens_with_spec = 5

step 1: computed 0 -> 5     (prefill 5 个 token；采样出 100)
step 2: computed 5 -> 6     (decode 1 个；输入是 100；采样出 101)
step 3: computed 6 -> 7     (decode 1 个；输入是 101；采样出 102)
step 4: computed 7 -> 7     (长度到 8？max_model_len 没到，但 max_tokens=3 已满足 -> finish)
```

### 例 2：chunked prefill（`max_num_batched_tokens = 2`）

```text
step 1: computed 0 -> 2     (只算 prompt[0:2]，不采样)
step 2: computed 2 -> 4     (prompt[2:4]，不采样)
step 3: computed 4 -> 5     (prompt[4:5]，采样出 100)
step 4: computed 5 -> 6     (decode 100，采样出 101)
```

**只有“追平”的那一步才采样**；中间的 chunk 不产生 token。nano 的 `ModelRunnerOutput`
对没追平的请求返回空 list，Scheduler 不能把它当成“生成了 0 个 token”。

### 例 3：prefix caching

```text
共享 system prompt 的 8 个 token，block_size=4
请求 A 先跑，块 0/1 已缓存
请求 B 的 num_computed 从 8 开始（而不是 0）
step 1: computed 8 -> prompt_end，只算 B 自己的后缀
```

注意 vLLM 的 `get_computed_blocks()` 会把命中长度限制在 `num_tokens - 1`：
**至少要重算最后一个 token 才能拿到 logits**。这个细节在 M07 会踩。

### 例 4：抢占

```text
running 里有 A、B；KV 不够了
Scheduler 选择牺牲 B（FCFS 下通常是最后进来的）
B.num_computed_tokens = 0
B.status = PREEMPTED
B 的 block 全部 free
B 被 prepend 回 waiting
下一步 B 从 0 重新 prefill（prefix cache 可能帮它少算一点）
```

这是 **recompute** 式抢占，不是 swap。vLLM V1 默认如此；优点是不用把 KV 换到 CPU，缺点是浪费算力。

### 例 5：spec decode（本课程不实现，但要知道）

```text
draft model 先猜 4 个 token: [100,101,102,103]
num_tokens_with_spec = 原长度 + 4
target model 一次前向验证这 4 个
接受 3 个 -> num_computed += 3，输出 3 个 token；第 4 个被拒绝，回滚
```

V1 的调度器不需要知道“草稿是怎么来的”，它只看到 `num_tokens_with_spec` 变大了。
这就是“追平”模型的通用性。

---

## 5. nanovllm 的分层与目录

我们按 vLLM 的**文件名和类名**复刻，但每个文件只保留最小实现：

```text
nanovllm/
├── README.md
├── ROADMAP.md
├── PROGRESS.md                  # 每个模块结束后更新
├── pyproject.toml
├── docs/
│   ├── 00_architecture.md       # 本文件
│   ├── M00_contracts.md
│   ├── M01_model.md
│   └── ...
├── nanovllm/
│   ├── config.py                # ModelConfig / CacheConfig / SchedulerConfig / EngineConfig
│   ├── sampling_params.py       # SamplingParams
│   ├── request.py               # Request / RequestStatus
│   ├── outputs.py               # ModelRunnerOutput / EngineCoreOutput / RequestOutput
│   ├── sched/
│   │   ├── output.py            # SchedulerOutput / NewRequestData / CachedRequestData
│   │   ├── request_queue.py     # FCFSRequestQueue
│   │   ├── scheduler.py
│   │   └── utils.py             # check_stop
│   ├── core/
│   │   ├── kv_cache_utils.py    # BlockHash / KVCacheBlock / FreeKVCacheBlockQueue
│   │   ├── block_pool.py        # BlockPool
│   │   └── kv_cache_manager.py  # KVCacheManager / KVCacheBlocks
│   ├── worker/
│   │   ├── block_table.py       # BlockTable
│   │   ├── input_batch.py       # CachedRequestState / InputBatch
│   │   ├── model_runner.py      # ModelRunner
│   │   └── gpu_worker.py        # GPUWorker：加载模型、测显存、建 KV cache
│   ├── attention/
│   │   ├── backend.py           # AttentionMetadata / AttentionBackend
│   │   ├── selector.py          # get_attn_backend()
│   │   ├── ops/
│   │   │   ├── cache.py         # reshape_and_cache / slot mapping
│   │   │   └── paged_attention.py  # Triton kernel
│   │   └── backends/
│   │       ├── naive.py         # gather + SDPA，正确性基准
│   │       └── triton.py        # 分页注意力 kernel
│   ├── models/
│   │   ├── qwen3.py             # Qwen3ForCausalLM
│   │   └── weight_loader.py     # safetensors 加载
│   ├── sample/
│   │   ├── sampler.py           # Sampler
│   │   └── metadata.py          # SamplingMetadata
│   ├── engine/
│   │   ├── core.py              # EngineCore
│   │   ├── llm_engine.py        # LLMEngine
│   │   ├── processor.py         # InputProcessor / OutputProcessor
│   │   ├── detokenizer.py       # IncrementalDetokenizer
│   │   └── async_llm.py         # M09
│   ├── executor/
│   │   ├── abstract.py          # Executor
│   │   └── uniproc_executor.py  # UniProcExecutor
│   └── entrypoints/
│       ├── llm.py               # LLM（离线高层 API）
│       └── openai_api.py        # M09
├── tests/
├── solutions/m00/               # 只有 M00 给参考解
├── scripts/
│   ├── check_model_parity.py
│   ├── download_model.py
│   └── verify_refs.py
└── benchmarks/
```

> 目录是**目标形态**，不是第一天就要全部建出来。M00 只建 `request.py`、`sched/output.py`、
> `utils/hashing.py` 和测试；后面的模块按需增加。

---

## 6. 冻结的公共接口

这一节是跨模块的“合同”。每个模块的 agent 都必须遵守；如果你确实要改，先改本文件并在
`PROGRESS.md` 里记一笔，否则下一个模块会对不上。

### 6.1 配置

```python
# nanovllm/config.py
@dataclass(frozen=True)
class ModelConfig:
    model: str
    dtype: torch.dtype = torch.bfloat16
    max_model_len: int = 4096
    block_size: int = 16
    seed: int = 0
    trust_remote_code: bool = False

@dataclass
class CacheConfig:
    block_size: int = 16
    gpu_memory_utilization: float = 0.85
    num_gpu_blocks: int | None = None
    enable_prefix_caching: bool = False

@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True

@dataclass
class EngineConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    scheduler_config: SchedulerConfig
    @classmethod
    def from_model(cls, model: str, **kwargs) -> "EngineConfig": ...
```

`ModelConfig.block_size` 与 `CacheConfig.block_size` 必须一致；M03 之后只保留一个来源。

### 6.2 请求

```python
# nanovllm/request.py
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

class Request:
    def __init__(self, request_id: str, prompt_token_ids: list[int],
                 sampling_params: SamplingParams, arrival_time: float = 0.0,
                 block_size: int = 16,
                 block_hasher: Callable[[Request], list[bytes]] | None = None,
                 client_index: int = 0): ...
    @property
    def all_token_ids(self) -> list[int]: ...
    @property
    def output_token_ids(self) -> list[int]: ...
    @property
    def num_tokens(self) -> int: ...
    @property
    def num_output_tokens(self) -> int: ...
    @property
    def num_tokens_with_spec(self) -> int: ...
    def append_output_token_ids(self, token_ids: int | list[int]) -> None: ...
    def update_block_hashes(self) -> None: ...
    def is_finished(self) -> bool: ...
    def get_finished_reason(self) -> "FinishReason | None": ...
```

`Request` 不持有 `block_ids`；物理块由 `KVCacheManager.req_to_blocks` 持有。
这样和 vLLM 一致，也避免“请求被抢占后 block id 残留”这类 bug。

### 6.3 调度契约

```python
# nanovllm/sched/output.py
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: list[int]
    num_computed_tokens: int
    num_output_tokens: int = 0

@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_block_ids: list[list[int] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] = field(default_factory=set)
    @classmethod
    def make_empty(cls) -> "SchedulerOutput": ...
    def validate(self, max_num_batched_tokens: int) -> None: ...
```

`validate()` 至少守住：

1. `total_num_scheduled_tokens == sum(num_scheduled_tokens.values())`
2. `total_num_scheduled_tokens <= max_num_batched_tokens`
3. 每个 `num_scheduled_tokens[req] > 0`
4. `finished_req_ids` 与 `num_scheduled_tokens` 不相交
5. `scheduled_new_reqs` 的 id 都在 `num_scheduled_tokens` 里
6. new / cached 的 id 不重复
7. cached 的四个 list 等长，且 id 都在 `num_scheduled_tokens` 里

### 6.4 KV 契约

```python
# nanovllm/core/kv_cache_utils.py
BlockHash = bytes
NONE_HASH: BlockHash

def hash_block_tokens(hash_function, parent_block_hash: BlockHash | None,
                      curr_block_token_ids: list[int],
                      extra_keys: tuple | None = None) -> BlockHash: ...
def get_request_block_hasher(block_size: int, hash_function) -> Callable[[Request], list[BlockHash]]: ...

@dataclass(eq=False)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHash | None = None
    prev_free: "KVCacheBlock | None" = None
    next_free: "KVCacheBlock | None" = None

# nanovllm/core/kv_cache_manager.py
@dataclass
class KVCacheBlocks:
    blocks: list[KVCacheBlock]
    def get_block_ids(self) -> list[int]: ...

class KVCacheManager:
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]: ...
    def allocate_slots(self, request: Request, num_new_tokens: int,
                       num_new_computed_tokens: int = 0,
                       new_computed_blocks: KVCacheBlocks | None = None
                       ) -> KVCacheBlocks | None: ...
    def free(self, request: Request) -> None: ...
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None: ...
    def get_block_ids(self, request_id: str) -> list[int]: ...
```

`get_computed_blocks()` 返回 `(命中的块, 命中的 token 数)`；没有缓存时返回 `(空, 0)`。
`allocate_slots()` 返回**本步新分配**的块；分配失败返回 `None`，由 Scheduler 触发抢占。
`free()` 必须把该请求所有块的 `ref_cnt` 减 1，归零的块按 LRU 顺序回到空闲队列。

### 6.5 注意力契约

```python
# nanovllm/attention/backend.py
@dataclass
class AttentionMetadata:
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor   # [num_reqs+1] int32
    seq_lens: torch.Tensor          # [num_reqs] int32
    block_table: torch.Tensor       # [num_reqs, max_num_blocks] int32
    slot_mapping: torch.Tensor      # [num_actual_tokens] int64
    causal: bool = True

class AttentionBackend(ABC):
    @abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                kv_cache: tuple[torch.Tensor, torch.Tensor],
                metadata: AttentionMetadata) -> torch.Tensor: ...
```

- `q`: `[num_tokens, num_heads, head_dim]`
- `k`/`v`: `[num_tokens, num_kv_heads, head_dim]`
- `kv_cache`: 该层的 `(k_cache, v_cache)`，形状 `[num_blocks, block_size, num_kv_heads, head_dim]`
- 返回 `[num_tokens, num_heads, head_dim]`

`forward()` **同时负责把 k/v 写进 cache**（通过 `slot_mapping`），再做 attention。
这样模型层不需要知道 cache 的布局。

### 6.6 ModelRunner / EngineCore

```python
# nanovllm/worker/model_runner.py
class ModelRunner:
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None: ...
    def sample_tokens(self) -> ModelRunnerOutput: ...

# nanovllm/executor/abstract.py
class Executor(ABC):
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None: ...
    def sample_tokens(self) -> ModelRunnerOutput: ...

# nanovllm/engine/core.py
class EngineCore:
    def add_request(self, request: Request) -> None: ...
    def abort_requests(self, request_ids: list[str]) -> None: ...
    def step(self) -> EngineCoreOutputs: ...
    def has_unfinished_requests(self) -> bool: ...
```

`execute_model()` 返回 `None` 表示“前向做完了，logits 暂存，等我调 sample_tokens”。返回
`ModelRunnerOutput` 表示“这一步没有需要采样的 token”（比如空 batch），EngineCore 不再调
`sample_tokens()`。这个约定是 M06 的核心，和 vLLM `EngineCore.step()` 完全一致。

### 6.7 前端

```python
# nanovllm/entrypoints/llm.py
class LLM:
    def __init__(self, model: str, **kwargs): ...
    def generate(self, prompts: str | list[str],
                 sampling_params: SamplingParams,
                 use_tqdm: bool = True) -> list[RequestOutput]: ...
    def stream(self, prompts: str | list[str],
               sampling_params: SamplingParams) -> Iterator[RequestOutput]: ...
```

`generate()` 的返回值顺序必须与输入 prompts 顺序一致；`stream()` 每步 yield 所有有增量的
请求（和 vLLM 的 `LLMEngine.step()` 语义一致）。

---

## 7. vLLM → nanovllm 映射表

| vLLM（@v0.32.0） | nanovllm | 说明 |
|---|---|---|
| `vllm/v1/request.py:60 Request` | `nanovllm/request.py` | 删多模态、LoRA、结构化输出、session |
| `vllm/v1/core/sched/output.py:232 SchedulerOutput` | `nanovllm/sched/output.py` | 删 encoder/KV connector/spec 字段；保留 new/cached 区分 |
| `vllm/v1/outputs.py:267 ModelRunnerOutput` | `nanovllm/outputs.py` | 只保留 req_ids/index/sampled/logprobs |
| `vllm/v1/core/sched/scheduler.py:591 schedule()` | `nanovllm/sched/scheduler.py` | 删优先级、encoder、connector、DP、streaming；保留 token budget/chunked/preempt |
| `vllm/v1/core/block_pool.py:136 BlockPool` | `nanovllm/core/block_pool.py` | 单 group、无 event/metrics/null block |
| `vllm/v1/core/kv_cache_manager.py:131 KVCacheManager` | `nanovllm/core/kv_cache_manager.py` | 单 group、无 sliding/Mamba/connector/COW |
| `vllm/v1/worker/gpu_model_runner.py:467 GPUModelRunner` | `nanovllm/worker/model_runner.py` | 删 CUDA graph/量化/LoRA/多模态/PP；保留 execute/sample 切分 |
| `vllm/v1/worker/gpu_input_batch.py:90 InputBatch` | `nanovllm/worker/input_batch.py` | 先 Python list，M10 再预分配 buffer |
| `vllm/v1/worker/block_table.py:49 BlockTable` | `nanovllm/worker/block_table.py` | 先 NumPy 算 slot mapping |
| `vllm/v1/attention/backend.py:59 AttentionBackend` | `nanovllm/attention/backend.py` | 先一个 `forward`，不做 metadata builder 继承树 |
| `vllm/v1/attention/backends/triton_attn.py:385 TritonAttentionImpl` | `nanovllm/attention/backends/triton.py` | 只保留 BF16、无 alibi/sliding/quant |
| `vllm/v1/attention/ops/triton_unified_attention.py:179 kernel` | `nanovllm/attention/ops/paged_attention.py` | 从“一 token 一 head 一 program”开始，不追求 vLLM 的 2D/3D 调度 |
| `vllm/v1/sample/sampler.py:22 Sampler` | `nanovllm/sample/sampler.py` | 先 greedy/temperature/top-k/top-p/penalty/logprobs |
| `vllm/v1/engine/core.py:115 EngineCore` | `nanovllm/engine/core.py` | 单进程同步循环 |
| `vllm/v1/engine/llm_engine.py:54 LLMEngine` | `nanovllm/engine/llm_engine.py` | 直接调 in-process EngineCore |
| `vllm/v1/engine/output_processor.py:681 OutputProcessor` | `nanovllm/engine/processor.py` | 单请求单输出，无 n>1/parallel sampling |
| `vllm/v1/engine/detokenizer.py:31 IncrementalDetokenizer` | `nanovllm/engine/detokenizer.py` | 先 full decode + delta，不做 DecodeStream |
| `vllm/model_executor/models/qwen3.py:280 Qwen3ForCausalLM` | `nanovllm/models/qwen3.py` | 纯 PyTorch，不 fused QKV，不做 TP |
| `vllm/v1/worker/gpu_worker.py:634 determine_available_memory` | `nanovllm/worker/gpu_worker.py` | 用 `torch.cuda.mem_get_info` 简化 profiling |

---

## 8. 有意删掉的东西（以及为什么）

| 删掉 | 原因 | 什么时候可以加回来 |
|---|---|---|
| 多模态 / encoder cache | 和语言模型主链路正交，代码量巨大 | 不需要 |
| LoRA / 量化 / KV connector | 属于部署优化，不是 V1 骨架 | 不需要 |
| MLA / Mamba / sliding window / hybrid allocator | 需要多 group KV，nano 先做单 group | 进阶 |
| TP / PP / DP / EP | 本机只有 1 张 GPU | M11 只做多进程 EngineCore，不做 TP |
| CUDA graph | 先保证正确性；图捕获会掩盖 shape bug | M10 |
| 异步调度 / batch queue | 先同步；M09 讲清 `execute/sample` 切分后再加 | M09/M10 |
| 结构化输出 / grammar | 需要额外依赖和 bitmask | 不需要 |
| spec decode | M05 的 `num_tokens_with_spec` 已留好位置 | 可选扩展 |
| ZMQ / msgspec / 共享内存 | 先 in-process；M11 用 `multiprocessing.Queue` 体验边界 | M11 |

**“等结构、缩规模”**：类名、文件位置、数据流尽量 1:1；删的是分支和优化，不是骨架。

---

## 9. 测试策略

### 9.1 三层测试

| 层 | 在哪跑 | 测什么 | 例子 |
|---|---|---|---|
| 单元测试 | CPU | 纯逻辑：哈希、块池、调度器、采样、detokenizer | `pytest tests/test_m03_block_pool.py` |
| GPU 对拍 | GPU | 和参考实现比数值：HF logits、naive attention vs Triton、连续 cache vs 分页 cache | `pytest tests/test_m04_paged_attention.py` |
| 端到端 | GPU | 真实模型、真实 tokenizer、连续批处理 | `python benchmarks/bench.py` |

### 9.2 对拍基准

1. **模型对拍**：`transformers` 的 `Qwen3ForCausalLM`（eager attention，bf16）作为 oracle；
   固定 prompt、固定权重，比较 logits / greedy token。
2. **注意力对拍**：M02 的连续 KV cache 实现作为 oracle；M04 的 naive paged 和 Triton paged
   都必须和它 `torch.allclose`。
3. **调度对拍**：M05 写一个 fake model runner（按固定规则吐 token），把调度 trace 打印出来，
   人工检查；M06 再用真实模型验证“批量输出 == 逐个输出”（greedy 下必须完全一致）。

### 9.3 测试的“金标准”

- 所有随机采样测试必须固定 seed；
- 所有 GPU 对拍必须用 `torch.testing.assert_close`，并写清 `atol/rtol`；
- 所有“性能提升”必须同时给出正确性断言，否则不算通过。

---

## 10. 怎么读 vLLM 这种“冗长”的代码

### 10.1 符号优先，行号第二

不要从文件第一行读。先 `grep` 符号，再读它前后 30~80 行：

```bash
cd /home/dictator/ai_infra/target/vllm-0.32.0
grep -n "class Scheduler" vllm/v1/core/sched/scheduler.py
grep -n "def schedule" vllm/v1/core/sched/scheduler.py
grep -n "def allocate_slots" vllm/v1/core/kv_cache_manager.py
```

### 10.2 只读“主路径”，跳过“分支”

vLLM 的 `schedule()` 有 900+ 行，但主路径只有：

```text
for running: 算 num_new_tokens -> allocate_slots -> 成功就记账 / 失败就抢占
for waiting: get_computed_blocks -> 算 num_new_tokens -> allocate_slots -> 进 running
组装 SchedulerOutput
```

其余全是：多模态、encoder cache、KV connector、DP、PP、spec decode、structured output、
sliding window、Mamba、priority、streaming input。**第一遍全部跳过**，在文档里会标“可跳过”。

### 10.3 用“数据”而不是“控制流”来理解

读一个函数时先问：

1. 它的输入是哪个数据结构？
2. 它修改了哪个字段？
3. 它的输出会变成谁的下一个输入？

例如 `allocate_slots()`：输入 `Request + num_new_tokens`，修改 `req_to_blocks` 和
`block.ref_cnt`，输出 `KVCacheBlocks`（给 `SchedulerOutput.new_block_ids`）。控制流再长，
数据流只有这一条。

### 10.4 每次只回答一个问题

不要“读完 Scheduler”。带着问题读：

- `num_computed_tokens` 在哪一行被加？
- 抢占时哪些字段被重置？
- `finished_req_ids` 是在哪里被填的？
- `SchedulerOutput` 是在哪一行组装的？

读完把答案写进 `PROGRESS.md` 或笔记里。下一篇模块文档会检查这些答案。

---

## 11. 术语表

| 术语 | 含义 |
|---|---|
| prefill | 一次算完（或分块算）prompt，建立 KV cache |
| decode | 每次算 1 个新 token，读已有 KV cache |
| chunked prefill | 把长 prompt 切成多个 chunk，分多个 step 算 |
| continuous batching | 每个 step 动态组 batch，请求可进可出 |
| paged KV cache | KV cache 切成固定大小 block，通过 block table 映射 |
| block table | 每个请求的逻辑块号 → 物理块号的映射表 |
| slot mapping | 本步每个 token 写到 KV cache 的哪个物理 slot |
| prefix caching | 相同前缀的请求复用已算好的 KV block |
| preemption | KV 不够时，把某个 running 请求的块全释放，之后重新 prefill |
| TTFT | time to first token，首 token 延迟 |
| TPOT | time per output token，decode 每 token 延迟 |
| GQA | grouped-query attention，多个 Q head 共享一个 KV head |
| RoPE | rotary position embedding，旋转位置编码 |

---

## 12. 给每个模块 agent 的“总规则”

1. **一次只教一个概念**；用户没复述清楚，不要往下走。
2. **不直接给完整实现**；先给接口、伪代码、失败测试，让用户自己写。
3. **每个任务都有验收命令**；没有测试通过就不算完成。
4. **引用必须带 `文件:行号`**；不确定就 `grep`，不要编。
5. **不读整个文件**；按模块文档给的区间读，超出的部分标记“可跳过”。
6. **结束前更新 `PROGRESS.md`，并建议一个 commit message**。
7. **不修改 `docs/00_architecture.md` 的冻结接口**；要改先和用户确认。

下一站：`docs/M00_contracts.md`。
