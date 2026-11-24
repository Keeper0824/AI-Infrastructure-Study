# M11 · 多进程 EngineCore（可选进阶）

> **模块目标**：把 M06 的单进程 `EngineCore` 挪到子进程，前端只通过队列收发消息；
> 亲手体验 V1 的**进程边界**、启动握手、背压和优雅退出。
> **预计工时**：3–4 小时　**依赖**：M06（最好先做完 M09）　**测试**：CPU 协议 + GPU 端到端
>
> **可选模块**：可以跳过 M11，直接做 M12。M12 的矩阵不依赖多进程；跳过时在
> `PROGRESS.md` 写 `M11 skipped`，M12 只报告单进程版本。
>
> **产出**：`EngineCoreProc` + `MPClient` + `InprocClient`；MP 与单进程 greedy 输出一致；
> `shutdown()` 后退出码 0；一张 IPC 开销表。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m11_protocol.py -q                 # CPU，无需 GPU
python -m pytest tests/test_m11_multiprocess.py -q -m gpu      # GPU 端到端
python benchmarks/bench_ipc.py --batch-sizes 1 8 --repeats 5   # IPC 开销表
```

五个能脱口而出的答案见 §10：为什么分进程、两个请求结构的边界、背压、`spawn`、abort 清理。

---

## 1. 为什么需要这个模块（10 分钟）

M06 的调用链是同步的：`LLMEngine.step()` 直接调用 `EngineCore.step()`，调度、GPU 前向、
采样、detokenize 都在一个 Python 进程里。小模型低并发时没问题，但有两个限制：CPU 调度
和 GPU 前向不能重叠；前端与引擎共享故障域。

vLLM V1 把引擎放进 `EngineCoreProc`：

```text
前端进程                                  引擎进程
LLMEngine / AsyncLLM  ──ADD/ABORT──►  input Q  ──►  EngineCoreProc
OutputProcessor       ◄──OUTPUT────  output Q ◄──  EngineCore
RequestState（detokenize）                         （Scheduler/Executor/KV）
```

收益：CPU 调度/输出处理与 GPU 前向重叠；前端崩溃不必然拖垮引擎；多前端可共享引擎。
代价：pickle、队列拷贝、上下文切换、启动/退出生命周期。**nano 只做“一个前端 + 一个引擎
进程 + 一个 GPU worker”，用 `multiprocessing.Queue`，不做 ZMQ/TP/PP/DP/共享内存**。
目的不是更快，而是量出进程边界的成本；0.6B 很小，IPC 在 step 中的占比会比大模型更明显。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`。行号已校验；漂了用 `grep -n` 重定位。

### 2.1 必读（按顺序）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/engine/core.py:1128` | `EngineCoreProc` | 继承 `EngineCore`；`__init__` 的 `input_queue`/`output_queue`（`:1147`/`:1148`） | ZMQ、DP、tensor IPC、fault tolerance |
| 2 | `vllm/v1/engine/core.py:1547` | `run_busy_loop()` | 三层：`_handle_shutdown` → `_process_input_queue` → `_process_engine_step` | DP 子类同名方法（`:2328`） |
| 3 | `vllm/v1/engine/core.py:1573` | `_process_input_queue()` | “有活可干”就退出输入循环；`get(block=...)` 的用法 | `aborts_queue`、idle callback |
| 4 | `vllm/v1/engine/core.py:1603` | `_process_engine_step()` | `step_fn()` 返回 `(outputs, model_executed)`；输出如何进队列 | `post_step` 的 spec 分支 |
| 5 | `vllm/v1/engine/core.py:1674` | `_handle_client_request()` | ADD/ABORT/UTILITY/EXECUTOR_FAILED 分发；ADD 的 payload | shutdown 拒绝、msgspec 转换 |
| 6 | `vllm/v1/engine/core_client.py:80` | `EngineCoreClient` | 抽象接口：`get_output`/`add_request`/`abort_requests`/`shutdown` | DP/elastic/fault 分支 |
| 7 | `vllm/v1/engine/core_client.py:583` | `MPClient` | 构造时等 READY；客户端为什么要单独线程收输出 | ZMQ、msgspec、DP 路由 |
| 8 | `vllm/v1/engine/core_client.py:914` | `SyncMPClient` | `get_output()`（`:989`）如何阻塞/抛异常；`add_request`（`:1021`）只发不等 | 后台 poller 细节 |
| 9 | `vllm/v1/engine/__init__.py:293` | `EngineCoreRequestType` | ADD/ABORT/UTILITY 的字节标记；vLLM **没有** SHUTDOWN，用 `WAKEUP` + shutdown state | START_DP_WAVE |
| 10 | `vllm/v1/engine/__init__.py:109` | `EngineCoreRequest` | 前端 → core 的字段；`client_index` 的用途 | 多模态、LoRA、embeds、session |
| 11 | `vllm/v1/engine/__init__.py:259` | `EngineCoreOutputs` | core → 前端的输出容器；`outputs` 是 list | stats、utility、wave 字段 |
| 12 | `vllm/v1/engine/utils.py:1198` | `launch_core_engines()` | 进程创建骨架：`CoreEngineProcManager`（`:237`）、`get_mp_context()`（`:256`）、`proc.start()`（`:329`） | Ray、elastic、NUMA |

> 小修正：早期笔记里的 `launch_engine_core` 在当前树实际是 `launch_core_engines`，
> 进程管理器叫 `CoreEngineProcManager`。引用以符号为准。

### 2.2 建议读（选读）

- `vllm/v1/engine/core.py:1391 · run_engine_core()`：子进程入口，先建 core 再进 loop。
- `vllm/v1/engine/__init__.py:73 · EngineCoreReadyResponse`：READY 带元数据，不只是布尔值。
- `vllm/utils/system_utils.py:164 · get_mp_context()`：默认 fork，CUDA 初始化后强制 spawn（`:122`）。
- `vllm/v1/executor/multiproc_executor.py:111 · MultiprocExecutor`：对照用，nano 不做 worker 组。

### 2.3 明确跳过

ZMQ、msgspec、shm_broadcast、tensor IPC、DP wave/coordinator、elastic EP、fault tolerance、
Ray、`MultiprocExecutor` 的 worker 组、`EngineCoreProc` 的完整生命周期。只复刻“消息循环”。

---

## 3. 核心概念

### 3.1 所有权：跨进程不传可变对象

| 东西 | 前端 | 引擎 |
|---|---|---|
| `request_id` / prompt token ids / `SamplingParams` | 持有（detokenize 用） | 持有（调度/采样用） |
| `Request` 对象 | **不持有** | 唯一持有者，只有它能改 `status`/`num_computed_tokens` |
| Scheduler / KV blocks | 不可见 | 持有 |
| 增量 token | 从 `EngineCoreOutputs` 收 | 由 `Scheduler.update_from_output` 产生 |

前端维护只读的 `RequestState`（prompt + 已生成 token + detokenizer），它不是调度器里的
`Request`。**跨进程只传不可变快照和 id。** M08 的 `OutputProcessor.add_request(request, prompt)`
只为取 prompt token ids / 采样参数；传完不要再持有或修改这个 `Request`。

### 3.2 消息协议

```python
class EngineCoreRequestType(enum.Enum):
    ADD = b"\x00"; ABORT = b"\x01"; SHUTDOWN = b"\x02"; UTILITY = b"\x03"

@dataclass(frozen=True)
class EngineCoreRequest:      # 前端 -> 引擎，必须可 pickle
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float
    client_index: int = 0

@dataclass(frozen=True)
class EngineCoreReady:        # 引擎 -> 前端，启动握手
    pid: int
    num_gpu_blocks: int

@dataclass(frozen=True)
class EngineCoreError:        # 引擎 -> 前端，异常回传；正常退出用 None 哨兵
    traceback: str
```

两点：① vLLM 没有 SHUTDOWN，它用 `WAKEUP` + `EngineShutdownState`；nano 显式 `SHUTDOWN` 更直观。
② READY/ERROR/None 走**输出队列**，`get_output()` 内部消费，只把 `EngineCoreOutputs` 返回前端。

### 3.3 启动握手：先 READY，再 ADD

```text
父进程                                  子进程
  │ start() ────────────────────────────► │ 建 EngineCore（加载模型/CUDA/KV）
  │ ◄──── READY(pid, num_gpu_blocks) ───── │ scheduler/executor 已可用
  │ add_request() ──────────────────────► │
```

模型加载要几十秒；先 ADD 会让消息排在无人消费的队列里，初始化失败时父进程还会等一个
永远不会来的输出。READY 同时承担“就绪”和“初始化成功”。超时建议 120s，超时杀进程并
抛 `TimeoutError`。

### 3.4 背压：有界输出队列

`output_queue = ctx.Queue(maxsize=8)`。引擎每步 `put`，前端每步 `get`。队列满 → 引擎阻塞，
GPU 空闲；如果前端此时等 `join()` → **死锁**。无界队列不会死锁，但前端慢时内存无限涨。

nano 的规则：**前端持续消费；`shutdown()` 期间也一边 drain 一边等**。背压不是加个
`maxsize`，而是消费者不能在生产者阻塞时停摆。

### 3.5 退出：状态机而不是一个 `terminate()`

```text
RUNNING ──shutdown()──► CLOSING ──put(SHUTDOWN)──► 引擎 break
   ▲                      │ drain output_queue        │ put(None)
   └──── join(timeout) ◄──┴──────────── 失败则 terminate + join，检查 exitcode
```

正常路径：`exitcode == 0`、`not is_alive()`、`shutdown()` 可重复调用。异常路径：父进程拿到
`EngineCoreError.traceback`，而不是挂死。

### 3.6 CUDA 与 fork：必须 spawn

`fork` 只复制调用线程，子进程继承已初始化的 CUDA context，再用 CUDA 会报错/死锁。vLLM 在
`cuda_is_initialized()` 时强制 spawn（`vllm/utils/system_utils.py:122 · _maybe_force_spawn`）。
nano 固定：

```python
ctx = multiprocessing.get_context("spawn")
input_queue = ctx.Queue(maxsize=64)
output_queue = ctx.Queue(maxsize=8)
proc = EngineCoreProc(config, input_queue, output_queue); proc.start()
```

spawn 会重新 import、重新初始化 torch/CUDA，所以 `EngineCoreProc.__init__` 只能存 `config`
和队列，**模型加载必须在 `run()` 里**。

### 3.7 什么能 pickle

| 对象 | 能过队列？ | 说明 |
|---|---|---|
| `list[int]` / `str` / `float` / `SamplingParams` | ✅ | nano 只传这些 |
| `EngineCoreRequest` / `EngineCoreOutputs` | ✅ | 只含标量和 list，不塞 tensor |
| `Request`（带 `_block_hasher` 闭包） | ❌ | 闭包不可 pickle，而且它不该跨进程 |
| `torch.Tensor` / `torch.Generator` | ❌/昂贵 | 会 pickle + 拷贝；nano 不传 |
| 模型 / tokenizer / KVCacheManager | ❌ | 只存在于引擎进程 |

`MPClient.add_request(request: Request)` 的签名只为和 `InprocClient` 同构；实现第一行就转成
`EngineCoreRequest`，然后立刻丢掉原对象。

## 4. 从 vLLM 到 nanovllm

| vLLM（@v0.32.0） | nanovllm | 保留 / 删除 / 简化 |
|---|---|---|
| `EngineCoreProc`（`core.py:1128`） | `engine/core_proc.py` | 保留 busy loop + 分发；删 ZMQ/DP/fault |
| `queue.Queue` + ZMQ socket | `multiprocessing.Queue` | 一个输入队列 + 一个有界输出队列 |
| `EngineCoreRequest`（`__init__.py:109`） | 同名 dataclass | 只留 id/prompt/sampling/arrival/client_index |
| `EngineCoreRequestType`（`:293`） | 同名 Enum | 只留 ADD/ABORT/SHUTDOWN/UTILITY |
| `EngineCoreReadyResponse`（`:73`） | `EngineCoreReady` | 只留 pid + num_gpu_blocks |
| `EngineCoreOutputs`（`:259`） | 复用 M00 定义 | 不加字段；READY/ERROR 是另一类消息 |
| `MPClient`/`SyncMPClient`（`core_client.py:583`/`:914`） | `MPClient` | 合并同步版；ZMQ 换 Queue；不需要输出线程 |
| `launch_core_engines` + `CoreEngineProcManager` | `MPClient.__init__` | 只要 `Process.start()` + 等 READY |
| `MultiprocExecutor`（`multiproc_executor.py:111`） | `UniProcExecutor`（引擎进程内） | 删 worker 进程组，不做 TP |
| `DPCoordinator`（`coordinator.py:23`） | 无 | 直接删 |

**本模块新增接口**（可选模块，不修改 `docs/00_architecture.md` §6.6 的 `EngineCore`）：

```python
# nanovllm/engine/core_proc.py
class EngineCoreProc(multiprocessing.Process):
    def __init__(self, config, input_queue, output_queue, core_factory=EngineCore): ...
    def run(self) -> None: ...
    def _handle_request(self, msg) -> None: ...
    def _process_step(self) -> None: ...

# nanovllm/engine/core_client.py —— 两个实现共享同一组签名
class EngineCoreClient(Protocol):
    def add_request(self, request: Request) -> None: ...
    def abort_requests(self, request_ids: list[str]) -> None: ...
    def get_output(self) -> EngineCoreOutputs: ...
    def has_unfinished_requests(self) -> bool: ...
    def shutdown(self) -> None: ...

class InprocClient: ...   # get_output() 内部调 EngineCore.step()
class MPClient: ...       # 队列实现；额外 call_utility() 仅供测试/调试
```

`LLMEngine(config, client=None, use_multiprocessing=False)`：默认 `InprocClient`，行为与 M06
一致；`LLM(..., use_multiprocessing=True)` 只是透传。这是唯一新增的用户可见开关。

## 5. 实现任务

### T1 · 消息协议（30 分钟）

文件：`nanovllm/engine/core_proc.py`（协议类型也放这里）；测试：`tests/test_m11_protocol.py`。

```python
def test_wire_request_drops_unpicklable_fields():
    req = Request("r1", [1, 2, 3], SamplingParams(max_tokens=4),
                  block_hasher=lambda r: [])
    wire = EngineCoreRequest.from_request(req)
    assert wire.request_id == "r1"
    assert pickle.loads(pickle.dumps(wire)).prompt_token_ids == [1, 2, 3]
    assert not hasattr(wire, "block_hasher")

def test_request_type_tags_are_unique():
    tags = [t.value for t in EngineCoreRequestType]
    assert len(tags) == len(set(tags))
```

验收：`python -m pytest tests/test_m11_protocol.py -q`。
坑：① 直接把 `Request` 放进队列；② 用字符串当消息类型；③ READY 忘了带 `num_gpu_blocks`。

### T2 · `EngineCoreProc` 循环（60 分钟）

文件：`nanovllm/engine/core_proc.py`。伪代码：

```python
def run(self):
    try:
        core = self._core_factory(self.config)        # 必须在子进程里建
        self.output_queue.put(EngineCoreReady(os.getpid(), core.num_gpu_blocks))
        while True:
            try:
                typ, payload = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                typ = None
            if typ is EngineCoreRequestType.SHUTDOWN:
                break
            if typ is not None:
                self._handle_request(core, typ, payload)
            if core.has_unfinished_requests():
                self._process_step(core)
    except BaseException:
        self.output_queue.put(EngineCoreError(traceback.format_exc())); raise
    finally:
        self.output_queue.put(None)
```

`timeout` 用于“没有新请求但还有未完成请求”时继续 decode，并定期检查退出；不要用
`Queue.empty()` 做控制流。`_process_step` 只 `put` 非空输出。

验收：`python -m pytest tests/test_m11_protocol.py -q -k "proc or loop"`。
用 `core_factory=FakeCore` 在 CPU 上测：每步追加固定 token，ABORT 后返回 `finish_reason=ABORT`。
坑：① 在 `__init__` 里建 `EngineCore`；② 没有 timeout，空闲忙等；③ 每轮都发空输出。

### T3 · `MPClient` 请求/输出（50 分钟）

文件：`nanovllm/engine/core_client.py`。

```python
def get_output(self) -> EngineCoreOutputs:
    self._wait_ready()                 # 内部消费 READY；超时/ERROR 直接抛
    while True:
        msg = self.output_queue.get(timeout=self._io_timeout)
        if isinstance(msg, EngineCoreOutputs):
            self._account_finished(msg)   # 只在这里减 _num_unfinished
            return msg
        if isinstance(msg, EngineCoreError):
            raise RuntimeError(msg.traceback)
        if msg is None:
            raise RuntimeError("engine exited unexpectedly")
```

`has_unfinished_requests()` 用前端本地计数：ADD +1，收到带 `finish_reason` 的输出 -1。
不要读引擎进程的 scheduler。

验收：`python -m pytest tests/test_m11_protocol.py -q -k client`。
坑：① 把 READY 当业务输出返回；② 每次 `get()` 新建 Queue；③ abort 时立刻 -1，提前退出。

### T4 · 启动握手与退出（40 分钟）

文件：`core_client.py` 的 `__init__` / `shutdown`。

验收：`python -m pytest tests/test_m11_multiprocess.py -q -m gpu -k "ready or shutdown or error"`。
覆盖：READY 超时；子进程初始化抛异常时父进程拿到 traceback；`shutdown()` 后
`proc.exitcode == 0` 且 `not proc.is_alive()`；重复调用安全。

坑：① 先 `join()` 再 drain，输出队列满时死锁；② 只 `terminate()` 不 `join()`，留僵尸；
③ 子进程正常退出也发 `EngineCoreError`。

### T5 · 接到 `LLMEngine`（40 分钟）

文件：`nanovllm/engine/llm_engine.py`、`nanovllm/entrypoints/llm.py`。

```python
class LLMEngine:
    def __init__(self, config, client=None, use_multiprocessing=False):
        self.client = client or (MPClient(config) if use_multiprocessing
                                 else InprocClient(config))
    def add_request(self, request_id, prompt, sampling_params):
        ...  # InputProcessor 分词；前端保留 RequestState
        self.output_processor.add_request(request, prompt)   # M08 的签名
        self.client.add_request(request)   # MPClient 内部转成 EngineCoreRequest
    def step(self):
        return self.output_processor.process_outputs(self.client.get_output())
```

验收：`python -m pytest tests/test_m11_multiprocess.py -q -m gpu -k matches_inproc`。
单请求、batch=2、chunked prefill 三种情况，MP 与 inproc 的 greedy token 逐 token 相同。
坑：① 前端又建一个 `Request` 并修改；② 忘了把 prompt 交给 `OutputProcessor`；
③ 异常退出时没关 client，子进程成孤儿。

### T6 · 并发与 abort（40 分钟）

文件：`tests/test_m11_multiprocess.py`。

```python
def test_abort_releases_kv(client, long_req):
    before = client.call_utility("num_free_blocks")
    client.add_request(long_req); client.get_output()      # 至少跑一步
    client.abort_requests([long_req.request_id])
    outputs = drain_until_finished(client)
    assert any(o.finish_reason is FinishReason.ABORT for o in outputs)
    assert client.call_utility("num_free_blocks") == before
```

还要有：两个不同 prompt 的并发请求输出不串；连续 add 100 个请求，全部 drain 后每个
request 的输出 token 数正确、没有重复 finish、没有丢请求。

验收：`python -m pytest tests/test_m11_multiprocess.py -q -m gpu -k "concurrent or abort or hundred"`。
坑：① `has_unfinished_requests()` 在最后一条输出入队前变假；② abort 输出被丢弃；
③ 前端不消费，输出队列满。

### T7 · IPC 开销 benchmark（30 分钟）

文件：`benchmarks/bench_ipc.py`（新脚本；结果写 `benchmarks/results/ipc.md`）。

- `add_to_first_ms`：add 到第一条输出（含排队 + 首次调度，不含模型加载）；
- `step_ms`：每次 `LLMEngine.step()` 的 wall time（前后 `torch.cuda.synchronize()`）；
- `e2e_ms`：add 到 finish。

batch=1/8 各 5 次，丢第 1 次（warmup），报中位数和 min。再加一个**纯队列微基准**：1000 个
`EngineCoreRequest` 塞进同一个 `ctx.Queue` 再取出，报告每条序列化+传输时间，用来解释 step
的固定开销。

验收：`python benchmarks/bench_ipc.py --batch-sizes 1 8 --repeats 5`，打印 markdown 表并写文件。
坑：① 把模型加载算进第一次 step；② 不同步 GPU；③ 两边 `max_tokens`/prompt 长度不同。

---

## 6. 测试与对拍

| 层 | 命令 | 关键断言 |
|---|---|---|
| 协议 | `pytest tests/test_m11_protocol.py -q` | wire 可 pickle、tag 唯一、READY 先于 ADD |
| 等价/并发 | `... test_m11_multiprocess.py -m gpu -k "inproc or concurrent"` | MP 与 inproc 相同；两请求不串 |
| abort/压力 | `... -k "abort or hundred"` | KV 恢复；100 请求不丢/不重 |
| 退出/异常 | `... -k "shutdown or error"` | exitcode 0、无僵尸；traceback 回传 |

失败现象 → 原因：第一次 `get_output()` 永久阻塞（没发 READY / READY 被丢）；step 越来越慢
最后卡死（前端不消费、队列满）；`Can't pickle local object`（Request/闭包进队列）；退出后
显存没释放（`terminate()` 杀进程）；MP 与 inproc 在某 token 分叉（前端多建了一份 `Request`）。

## 7. 常见坑

1. **fork 后碰 CUDA**：固定 spawn。
2. **输出队列满死锁**：前端持续消费；shutdown 时也要 drain。
3. **子进程异常没回传**：用 `EngineCoreError`，不要只打 stderr。
4. **`Request` 不可 pickle**：`_block_hasher` 是闭包，`generator` 是 torch 对象。
5. **退出没 join**：僵尸进程；`terminate()` 后不 join，显存不释放。
6. **顺序反了**：ADD 早于 READY，可能丢请求或挂死；多前端还要按 `client_index` 路由，nano 只做单前端。
7. **`Queue.empty()` 当控制流 / 脚本没有 `__main__` 守卫**：前者不可靠，后者 spawn 会递归。

## 8. 完成定义（DoD）

- [ ] 协议测试全绿（CPU）；
- [ ] MP 与 inproc 在单请求 / batch=2 / chunked prefill 下 greedy 逐 token 一致；
- [ ] 两个并发请求不串；abort 后 `num_free_blocks` 恢复且前端收到 ABORT；
- [ ] 100 个连续请求不丢、不重、不提前结束；
- [ ] `shutdown()` 后 `exitcode == 0`、无僵尸、可重复；初始化失败时父进程抛异常；
- [ ] `benchmarks/results/ipc.md` 有 batch=1/8 的额外延迟表；`PROGRESS.md` 已更新；
- [ ] commit：`M11: multiprocess engine core (queues + ready handshake)`。

---

## 9. 交接：给 M12

**新增**：`engine/core_proc.py`、`engine/core_client.py`、`tests/test_m11_protocol.py`、
`tests/test_m11_multiprocess.py`、`benchmarks/bench_ipc.py`。**修改**：`engine/llm_engine.py`
与 `entrypoints/llm.py`（可选透传）。

**公开 API**：`LLM(..., use_multiprocessing=True)`；`MPClient` 的五个方法。
`EngineCore.step()`、`SchedulerOutput`、`EngineCoreOutputs` **均未改**。M09 的 `_output_loop`
只需把 `run_in_executor(None, engine_core.step)` 换成 `...get_output`，其余方法不变；
不需要另写 asyncio client。

**已知限制**：单前端、单引擎进程；无 TP/PP/DP/worker 组；队列传输，无 shm/ZMQ/msgspec；
无心跳，僵死只能 `join(timeout)` + `terminate()`。

**M12 入口**：`docs/M12_capstone.md`。跳过 M11 时在 `PROGRESS.md` 写明，M12 忽略 IPC 一节。

## 10. 自测题

1. 为什么 V1 要把 EngineCore 放子进程？代价是什么？
2. `EngineCoreRequest` 和 `SchedulerOutput` 的边界在哪？为什么不是同一个东西？
3. 输出队列满时会发生什么？怎么避免死锁？
4. 为什么子进程要用 spawn 而不是 fork？
5. 前端 abort 一个请求，引擎里要做哪些清理？哪些在 worker 侧？

## 11. 教练流程（给 agent 用）

概念 20min → 读 `EngineCoreProc` 30min → 设计协议 20min → 写子进程 60min →
写 client 50min → 联调/退出 40min → benchmark 20min → 复盘 15min。

先“启动 → 一个请求 → 退出”，再并发/abort；先跑 inproc 回归再切 MP；卡住先看
`proc.is_alive()`、`proc.exitcode`、输出队列有没有人消费。

**禁止**：给完整 `run()` 实现；一次改 3 个文件；跳过 READY 直接 add；建议安装 vLLM。
