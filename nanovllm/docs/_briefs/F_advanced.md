# 文档作者 brief F：M11 多进程 EngineCore + M12 Capstone

## 通用要求

工作区：`/home/dictator/ai_infra/nanovllm`

先读：

1. `docs/00_architecture.md`（§1–§6、§8）
2. `docs/00_facts.md`（环境、行号、写作约定）
3. `docs/M00_contracts.md`（风格）
4. `docs/_template.md`
5. `ROADMAP.md`
6. `docs/_briefs/D_engine.md` 的 M06 部分（EngineCore/LLMEngine）
7. `docs/_briefs/E_frontend.md` 的 M09 部分（AsyncLLM）

只写：

- `docs/M11_multiprocess.md`
- `docs/M12_capstone.md`

不要改代码/其他文档。每个文件 320–450 行。中文。

M11 是**可选进阶**，M12 是**收尾**。文档要明确“可以跳过 M11，直接做 M12”。

---

## M11：多进程 EngineCore

### 目标

把 M06 的单进程 `EngineCore` 挪到子进程：

- 前端进程：`LLMEngine` / `AsyncLLM` + `MPClient`；
- 引擎进程：`EngineCoreProc` 循环：收请求 → schedule/execute/sample/update → 发输出；
- 用 `multiprocessing.Queue`（或 `Pipe`）传 `EngineCoreRequest` / `EngineCoreOutputs`；
- 支持 add / abort / 优雅退出；
- 测量 IPC 开销。

不做：TP/PP/DP、ZMQ、msgspec、共享内存、CUDA graph 跨进程、vLLM 的完整
`EngineCoreProc` 生命周期。只做“一个前端 + 一个引擎进程 + 一个 GPU worker”。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/engine/core.py:1128` | `EngineCoreProc` | 继承 EngineCore，多了 input/output 队列 |
| `:1547-1605` | `run_busy_loop()` | `_process_input_queue` → `_process_engine_step` |
| `:1603-1674` | `_process_engine_step()` | schedule/execute/sample/update 的封装 |
| `:1674-1800` | `_handle_client_request()` | ADD / ABORT / UTILITY 消息 |
| `vllm/v1/engine/core_client.py` | `MPClient` / `AsyncMPClient` | 前端如何发请求、收输出 |
| `vllm/v1/engine/__init__.py:293` | `EngineCoreRequestType` | ADD/ABORT/... 的字节标记 |
| `vllm/v1/engine/__init__.py:109` | `EngineCoreRequest` | 前端→core 的请求结构 |
| `vllm/v1/engine/__init__.py:259` | `EngineCoreOutputs` | core→前端 |
| `vllm/v1/engine/utils.py` | `launch_engine_core` / 进程启动 | 只看进程创建/握手骨架 |
| `vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor` | worker 进程组；nano 不做，只对比 |
| `vllm/v1/engine/coordinator.py` | `DPCoordinator` | 可选，理解 DP 才需要 |

**明确跳过**：ZMQ、msgspec、shm_broadcast、tensor IPC、DP wave、elastic EP、
fault tolerance、Ray。

### 必须讲清的概念

1. **为什么要分进程**：调度/输出处理（CPU）和 GPU 前向可以重叠；前端崩溃不拖垮引擎；
   多前端共享一个引擎。但 IPC 有成本，nano 用 Queue 体验边界。
2. **消息类型**：ADD / ABORT / SHUTDOWN；输出是 `EngineCoreOutputs`。
3. **所有权**：`Request` 对象只在引擎进程里；前端只持有 `request_id` 和
   `RequestState`（prompt/detokenizer）。
4. **启动握手**：子进程先建 EngineCore（加载模型、KV cache），再发 READY；前端收到
   READY 才能 add。
5. **背压**：输出队列满时引擎会阻塞；前端必须持续消费。
6. **退出**：发 SHUTDOWN → join → 关闭队列；处理子进程异常/僵死。
7. **CUDA + fork**：用 `multiprocessing.get_context("spawn")`，不要在 fork 后碰 CUDA；
   子进程里再初始化 torch/CUDA。
8. **序列化**：`Request` 里有 `SamplingParams` / list，pickle 没问题；不要传 torch tensor
   （除非用 shared memory）。nano 只传 token ids。

### 必须实现的接口

```python
# nanovllm/engine/core_proc.py
class EngineCoreProc(multiprocessing.Process):
    def __init__(self, config, input_queue, output_queue): ...
    def run(self) -> None: ...
    def _handle_request(self, msg) -> None: ...
    def _process_step(self) -> None: ...

# nanovllm/engine/core_client.py
class MPClient:
    def __init__(self, config): ...
    def add_request(self, request: Request) -> None: ...
    def abort_requests(self, ids: list[str]) -> None: ...
    def get_output(self) -> EngineCoreOutputs: ...
    def shutdown(self) -> None: ...

# nanovllm/engine/llm_engine.py
class LLMEngine:
    # 通过 client 抽象切换 InprocClient / MPClient
```

### 实现任务

T1 定义消息协议（ADD/ABORT/SHUTDOWN/READY）；T2 `EngineCoreProc` 循环；
T3 `MPClient` 请求/输出；T4 启动握手/退出；T5 接到 `LLMEngine`；
T6 并发/abort 测试；T7 IPC 开销 benchmark。

### 测试与对拍

- 子进程生成结果与单进程完全一致（greedy）；
- 2 个并发请求不串；
- abort 后引擎释放 KV，`num_free_blocks` 恢复；
- 连续 add 100 个请求，输出不丢/不重；
- 关闭：`shutdown()` 后进程退出码 0，无僵尸；
- IPC 开销：对比单进程，报告每 step 的额外延迟（batch=1/8）。

### 常见坑

- fork 后 CUDA 初始化报错 → 用 spawn；
- 输出队列满导致死锁 → 前端持续消费 / 有界队列 + 背压；
- 子进程异常没有传回前端 → 加错误消息/心跳；
- `Request` 里有不可 pickle 的字段（generator/tensor）→ 只传必要字段；
- 退出时没 join，僵尸进程；
- 顺序：先发 READY 再收请求；否则前端 add 丢失；
- 多前端共享一个引擎时，输出路由错（nano 只做单前端）。

### 自测题

1. 为什么 V1 要把 EngineCore 放子进程？代价是什么？
2. `EngineCoreRequest` 和 `SchedulerOutput` 的边界在哪？为什么不是同一个东西？
3. 输出队列满时会发生什么？怎么避免死锁？
4. 为什么子进程要用 spawn 而不是 fork？
5. 前端 abort 一个请求，引擎里要做哪些清理？

### 教练流程

概念 20min → 读 EngineCoreProc 30min → 设计消息协议 20min → 写子进程 60min →
写 client 50min → 联调/退出 40min → benchmark 20min → 复盘 15min。
先做“启动 → 一个请求 → 退出”，再做并发/abort；不要一上来做 DP。

---

## M12：Capstone：对拍、性能报告、代码地图

### 目标

把前 11 个模块收束成一份可展示的成果：

1. **正确性矩阵**：HF / 连续 cache / paged naive / paged Triton / 批量 vs 逐个 /
   prefix cache on/off；
2. **性能矩阵**：TTFT / TPOT / throughput，batch size、prompt length、output length、
   graph on/off、prefix cache on/off；
3. **代码地图**：每个 nano 文件 ↔ vLLM 文件，列出保留/删除/简化；
4. **REPORT.md**：结果、瓶颈、学到的 5 件事、后续可做的 3 件事；
5. **收尾**：测试全绿、README 更新、tag `v1.0`。

### 必须回读的 vLLM 代码

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/engine/core.py:657` | `EngineCore.step()` | 10 行主循环，对照 nano |
| `vllm/v1/core/sched/scheduler.py:591` | `Scheduler.schedule()` | 只看主路径，确认 nano 的简化 |
| `vllm/v1/core/kv_cache_manager.py:371` | `allocate_slots()` | 对照 nano 的 10 步算法 |
| `vllm/v1/worker/gpu_model_runner.py:1920` | `_prepare_inputs()` | 对照 nano 的 10 行流水线 |
| `vllm/v1/attention/ops/triton_unified_attention.py:179` | kernel | 对照 nano 的简化 kernel |
| `vllm/v1/engine/llm_engine.py:319` | `LLMEngine.step()` | 对照 nano 前端 |

### 必须讲清的概念

1. **公平 benchmark**：warmup、固定 prompt 集、固定 output 长度、同步计时、
   同一 GPU、同一 dtype；报告 min/median/p95。
2. **TTFT / TPOT / throughput 的定义**：
   - TTFT：请求到首 token；
   - TPOT：首 token 后每个 token 的平均时间；
   - throughput：总 output tokens / 总时间（或 requests/s）。
3. **Amdahl 定律**：nano 的瓶颈可能不在 attention，而在 Python 调度/采样/ detokenize；
   用 profiler 找。
4. **vLLM 对比的公平性**：版本、torch、CUDA、模型、batch、采样参数一致；vLLM 有
   CUDA graph/FlashAttention/continuous batching 优化，nano 不追求赢，只解释差距。
5. **代码地图**：不是“文件列表”，而是“每个 vLLM 机制在 nano 的哪一行、删了什么”。

### 实现任务

T1 正确性矩阵脚本；T2 性能矩阵脚本；T3 profile（torch profiler / Nsight）；
T4 代码地图；T5 REPORT.md；T6 README/文档收尾；T7 tag。

### 测试与对拍

- 所有 M00–M10 测试全绿；
- 正确性矩阵每一格都有 `assert`；
- 性能矩阵每个数字有 3 次重复 + 中位数；
- 报告里每个结论都有数据支撑；
- `git status` 干净，tag 打上。

### 常见坑

- benchmark 没 warmup，第一轮包含编译/加载；
- 没同步 GPU，计时偏小；
- 不同 batch 用了不同 output 长度；
- 把 vLLM 的 TP=1 和 nano 的 batch=1 直接比，不公平；
- 报告只写“快了多少”，没写“为什么”；
- 代码地图漏了简化项，读者以为 nano 支持全部功能。

### 自测题

1. TTFT 和 TPOT 分别反映什么瓶颈？
2. 为什么 benchmark 必须 warmup + 同步？
3. 你的 nano 最大的性能瓶颈在哪？怎么证明？
4. 如果要把 nano 的性能提高 2 倍，你先改哪三处？
5. 代码地图里，哪三个 vLLM 机制是你删得最狠的？为什么可以删？

### 教练流程

先跑全量测试 → 写正确性矩阵 60min → 写 benchmark 60min → profile 40min →
写报告 60min → 代码地图 30min → 收尾 20min。
不要在这个模块加新功能；只做验证、测量、写作。

---

## 两篇文档的衔接

- M11 是可选的；如果跳过，M12 直接基于单进程版本做报告。
- M12 不引入新公共接口；只读、测、写。
- M12 的 REPORT.md 应该引用 M04/M06/M10 的 benchmark 脚本，不要重复造轮子。
- 完成后，`PROGRESS.md` 写“课程完成”，并给出下一步建议（TP/spec/quant）。

## 交付前检查

- [ ] 两个文件 320–450 行；
- [ ] M11 明确“可选”，有 spawn/队列/退出/背压的坑；
- [ ] M12 有正确性矩阵、性能矩阵、代码地图、报告模板；
- [ ] 不引入新功能；
- [ ] 没有建议读完整 vLLM 进程/调度代码。
