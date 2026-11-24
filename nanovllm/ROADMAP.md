# ROADMAP · 13 个模块

> 每个模块对应一次独立的 agent 对话窗口。模块文档放在 `docs/MXX_*.md`。
> 顺序不是随便排的：每个模块的“出口”就是下一个模块的“入口”。

---

## 1. 总览

| # | 模块 | 一句话目标 | 主要 vLLM 阅读 | 产出 | 预计 |
|---|---|---|---|---|---|
| **M00** | 全景与契约 | 搞懂 V1 的三层数据契约，实现 `Request` / `SchedulerOutput` / 块哈希 | `request.py`、`sched/output.py`、`kv_cache_utils.py` | CPU 测试全绿；能画出 step 时序 | 2–3h |
| **M01** | Qwen3 前向 | 用纯 PyTorch 写 Qwen3，加载 safetensors，和 HF logits 对拍 | `models/qwen3.py`、`qwen2.py`、`layers/layernorm.py`、RoPE | `check_model_parity.py` 通过 | 3–4h |
| **M02** | 朴素 KV cache | 连续 KV cache + 自回归生成，第一次吐出完整文本 | `gpu_model_runner._prepare_inputs`、HF `past_key_values` | CLI 生成；HF greedy 对拍 | 3–4h |
| **M03** | Paged KV | `BlockPool` + `KVCacheManager` + `BlockTable` + slot mapping | `block_pool.py`、`kv_cache_manager.py`、`block_table.py` | 块池/分配/释放测试；KV 读写正确 | 4–5h |
| **M04** | Paged Attention | 注意力后端抽象 + naive/Triton 两个分页后端 | `attention/backend.py`、`triton_attn.py`、`triton_unified_attention.py` | 三路对拍（连续/naive/Triton） | 4–5h |
| **M05** | Scheduler | token 预算、连续批处理、chunked prefill、抢占 | `scheduler.py` 主路径、`request_queue.py`、`utils.check_stop` | CPU 调度 trace 测试 | 4–5h |
| **M06** | Engine 端到端 | `InputBatch` + `ModelRunner` + `EngineCore` + `LLM.generate` | `gpu_model_runner.py`、`gpu_input_batch.py`、`engine/core.py` | 8 路并发生成；批量==逐个 | 5–6h |
| **M07** | Prefix Caching | 链式哈希、前缀命中、LRU 驱逐、引用计数 | `block_pool.cache_full_blocks`、`get_computed_blocks` | 命中率/TTFT 对比；正确性测试 | 3–4h |
| **M08** | 采样与输出 | temperature/top-k/top-p/penalty/logprobs + 增量 detokenize + 流式 | `sample/sampler.py`、`detokenizer.py`、`output_processor.py` | 采样测试；流式输出；stop 条件 | 3–4h |
| **M09** | 服务化 | `AsyncLLM` + OpenAI 兼容 HTTP/SSE + abort | `async_llm.py`、`entrypoints/openai/` | curl 流式；并发/中断测试 | 4–5h |
| **M10** | 性能 | decode CUDA graph、pinned memory、benchmark | `capture_model`、`profile_run`、`cudagraph_dispatcher` | graph on/off 对拍；性能表 | 3–4h |
| **M11** | 多进程 EngineCore | 把 EngineCore 挪到子进程，理解 V1 进程边界 | `EngineCoreProc`、`core_client.py`、`multiproc_executor.py` | 子进程生成；干净退出 | 3–4h |
| **M12** | Capstone | 全量对拍、性能报告、代码地图、收尾 | 回读 `EngineCore.step` + `Scheduler.schedule` | `REPORT.md` + tag | 2–3h |

总计约 **45–60 小时**。如果每天 2 小时，大约 4–6 周；如果全天投入，1–2 周。

---

## 2. 依赖图

```text
M00 契约
 │
 ├──► M01 模型前向
 │      │
 │      └──► M02 朴素 KV + 生成
 │             │
 │             ├──► M03 Paged KV
 │             │      │
 │             │      └──► M04 Paged Attention
 │             │             │
 │             │             ├──► M06 Engine 端到端 ◄── M05 Scheduler
 │             │             │        │
 │             │             │        ├──► M07 Prefix Caching
 │             │             │        ├──► M08 采样/输出
 │             │             │        │      │
 │             │             │        │      └──► M09 服务化
 │             │             │        │             │
 │             │             │        │             └──► M10 CUDA Graph
 │             │             │        │                    │
 │             │             │        │                    └──► M11 多进程
 │             │             │        │                           │
 │             └─────────────┴────────┴───────────────────────────┴──► M12 Capstone
```

**硬依赖**：

- M01 依赖 M00 的 `Request`/`SamplingParams`（但模型前向本身可以先不接 request）；
- M02 依赖 M01；
- M03 依赖 M00 的哈希、`Request`；
- M04 依赖 M02（连续 cache 作为 oracle）和 M03（block table）；
- M05 依赖 M00 的 `SchedulerOutput`、M03 的 `KVCacheManager` 接口（可以先 mock）；
- M06 依赖 M02/M03/M04/M05；
- M07 依赖 M03/M06；
- M08 依赖 M06；
- M09 依赖 M08；
- M10 依赖 M06；
- M11 依赖 M09（或至少 M06）；
- M12 依赖全部。

**软依赖**：M07 和 M08 可以交换顺序；M10 可以在 M09 前做；M11 可以跳过。

---

## 3. 里程碑 Demo

每完成一个阶段，你应该能跑出一个“可见”的东西。不要等到最后才第一次生成文本。

| 里程碑 | 完成后你能跑 | 证明什么 |
|---|---|---|
| **A. 契约**（M00） | `pytest tests -q` 全绿；手推 2 请求调度 trace | 你理解了 V1 的数据流 |
| **B. 第一次生成**（M01–M02） | `python -m nanovllm.generate --prompt "..."` 输出连贯文本 | 模型、权重、KV、采样都通了 |
| **C. 分页注意力**（M03–M04） | 同一 prompt，连续/naive/Triton 三路 logits 一致 | PagedAttention 真的工作了 |
| **D. 连续批处理**（M05–M06） | `LLM.generate([p1..p8])`，吞吐明显高于逐个跑 | Scheduler + Runner 集成成功 |
| **E. 生产特性**（M07–M09） | 前缀缓存 TTFT 下降；`curl` 流式输出 | 你有了一个能用的迷你 vLLM |
| **F. 性能/边界**（M10–M11） | graph 加速；前端/引擎分进程 | 你理解了 V1 的性能与进程模型 |
| **G. 收尾**（M12） | `REPORT.md` + `git tag v1.0` | 可以给别人讲这门课了 |

---

## 4. 每个模块的固定交付物

每个模块结束时，仓库里必须有：

1. **代码**：该模块的 `nanovllm/...` 文件 + 至少一个测试文件；
2. **测试**：`pytest tests/test_mXX_*.py` 全绿（GPU 测试可用 `-m gpu` 标记）；
3. **文档**：`PROGRESS.md` 更新（完成内容、已知偏差、下一个模块的入口）；
4. **提交**：`git commit -m "MXX: <一句话>"`；
5. **口头验收**：能不看代码回答该模块文档最后的 5 个自测题。

> 不要在一个 commit 里混两个模块。每个模块一个 commit，出问题容易回滚。

---

## 5. 如果时间不够：三条路径

### 5.1 最短可用路径（约 20h）

`M00 → M01 → M02 → M05 → M06`（跳过 paged KV/attention，先用连续 cache 跑连续批处理）。

- 优点：最快看到“能用的引擎”；
- 缺点：没有 PagedAttention，学不到 vLLM 的核心内存管理；
- 适合：只想理解调度和引擎结构。

### 5.2 标准路径（约 35h）

`M00 → M01 → M02 → M03 → M04 → M05 → M06 → M08`。

- 覆盖 PagedAttention + 连续批处理 + 采样/输出；
- 不做 prefix caching、服务化、CUDA graph；
- 适合：想真正理解 vLLM 核心。

### 5.3 完整路径（约 50h+）

全部 13 个模块。M10–M12 是进阶，卡住可以跳过，不影响主线。

---

## 6. 风险与对策

| 风险 | 症状 | 对策 |
|---|---|---|
| 模型权重加载对不上 | 输出乱码 / logits 全 NaN | 先跑 M01 的 parity；逐层对比 HF hidden states；检查 tied embeddings |
| RoPE 位置错 | 短 prompt 正常，长 prompt 崩 | 检查 `positions` 是绝对位置，不是 chunk 内偏移 |
| KV cache 写错 slot | 输出逐渐变成重复/乱码 | 用 M03 的 slot mapping 单测；先 naive gather 对拍 |
| Triton kernel 编译失败 | `CompilationError` / 越界 | 先用 `BLOCK_SIZE=16`、`head_dim=128` 固定常量；打印 grid |
| 调度器死循环 | `generate()` 不结束 | 打印每步 `num_scheduled_tokens` 和 `num_computed_tokens`；检查 finish 条件 |
| prefix cache 命中错块 | 第二个请求输出和第一个不同 | 检查链式哈希的 parent；检查 `num_tokens - 1` 的限制 |
| 显存 OOM | 大 prompt / 多并发 | 降 `max_model_len`、`gpu_memory_utilization`；先跑小模型 |
| CUDA graph 结果错 | graph on 输出乱码 | 先保证 eager 正确；检查静态 buffer 的拷贝和 shape padding |

---

## 7. 版本与行号

- 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`
- 所有引用格式：`vllm/v1/...py:行号 · 符号`
- 行号漂了怎么办：

```bash
cd /home/dictator/ai_infra/target/vllm-0.32.0
grep -n "def schedule" vllm/v1/core/sched/scheduler.py
grep -n "class SchedulerOutput" vllm/v1/core/sched/output.py
```

- 本课程不要求安装 vLLM（它要求 torch 2.13，和本机 2.11 冲突）；只读源码。
- M12 如果想跑 vLLM 对比，建议单独建 venv，避免污染主环境。

---

## 8. 开始之前

1. 读 `docs/00_architecture.md` 的 §1–§6；
2. 读 `docs/M00_contracts.md`；
3. 确认环境（conda 环境 `vllm_learn`，不是 base）：

```bash
conda activate vllm_learn
cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py
python --version                 # 3.11.15
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests -q        # 31 failed, 3 passed（M00 的起点）
```

如果 `doctor.py` 报没有 pytest：

```bash
python -m pip install pytest
```

4. 打开一个新的 agent 窗口，把 `README.md` 里的“用户提示词模板”贴进去。

下一步：`docs/M00_contracts.md`。
