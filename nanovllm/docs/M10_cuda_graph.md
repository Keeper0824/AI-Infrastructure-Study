# M10 · CUDA Graph 与 decode 性能

> **模块目标**：在 M06 的 decode 路径上做 **decode-only CUDA graph**：预分配静态输入 buffer，
> 按 batch size bucket 捕获，replay 时只拷贝输入；prefill / chunked 继续 eager。
> **预计工时**：3–4 小时　**依赖**：M06（ModelRunner）、M04（Triton backend）　**测试**：GPU
>
> **产出**：`pytest tests/test_m10_cuda_graph.py -q` 全绿；graph on/off 对拍；
> `benchmarks/bench_m10_decode.py` 的 TTFT / TPOT / throughput 表。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m10_cuda_graph.py -q
python benchmarks/bench_m10_decode.py --batch-size 1 8 32 --graph on
```

以及能不看代码回答 §10 的五个问题。

---

## 1. 为什么需要这个模块（10 分钟）

decode 的一步里，每个请求只算 1 个 token，但模型仍有 28 层，每层要做 RMSNorm、QKV
projection、RoPE、attention、O projection、MLP、残差。**GPU 计算量很小，CPU 侧要发几十到
上百个 kernel launch**；当 batch 不大时，launch 开销和 Python 开销会占 decode 时间的很大
一部分。

CUDA graph 把这些 launch 一次性录下来，之后一次 `replay()` 重放。关键约束是：graph 记录的
是**地址和 Python 标量**，不是“这一 step 的值”。所以：

- decode 适合：每步 shape 几乎固定（batch × 1 token），变的只有 `seq_lens`、`block_table`、
  `slot_mapping` 这些**张量内容**；
- prefill 不适合：prompt 长度、chunk 长度每步都变，shape 不固定；nano 继续 eager。

本模块只做“固定 batch bucket 的 decode graph”，不做 piecewise/full graph，不做
torch.compile，不改 M04 的 kernel 接口，也不改 §6.6 的 `ModelRunner.execute_model` /
`sample_tokens` 签名。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`
> `gpu_model_runner.py` 有 7200+ 行，**不要通读**；只读下表锚点前后 30–80 行。

### 2.1 必读（按顺序）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/worker/gpu_model_runner.py:467` | `GPUModelRunner` | `requests` / `input_batch` / `kv_caches` 字段；`cudagraph_dispatcher` 在 `:873` | 多模态 / LoRA / PP |
| 2 | `vllm/v1/worker/gpu_model_runner.py:771` | `input_ids` / `positions` / `query_start_loc` / `seq_lens` buffer | “persistent buffers for CUDA graphs” 这一段的分配方式 | 后面的 spec / mamba buffer |
| 3 | `vllm/v1/worker/gpu_model_runner.py:1920` | `_prepare_inputs()` | 每步哪些张量是“输入”；`query_start_loc` / `seq_lens` / slot mapping 怎么来 | spec / mrope / async |
| 4 | `vllm/v1/worker/gpu_model_runner.py:5508` | `_dummy_run()` | dummy batch 怎么构造、warmup 和 capture 共用什么 | LoRA / 多模态 / EPLB |
| 5 | `vllm/v1/worker/gpu_model_runner.py:6133` | `profile_run()` | 峰值显存 profiling；graph 显存为什么要另算 | 多模态 encoder |
| 6 | `vllm/v1/worker/gpu_model_runner.py:6482` | `capture_model()` | 捕获顺序、同步点、大 bucket 先、capture 后测显存 | encoder graph / profiler |
| 7 | `vllm/v1/cudagraph_dispatcher.py:15` | `CudagraphDispatcher` | batch descriptor / key / `dispatch()` 的职责 | LoRA / piecewise / full |
| 8 | `vllm/v1/cudagraph_dispatcher.py:71` | `_compute_bs_to_padded_graph_size()` | batch size → 最近 bucket 的映射 | compile sizes 校验 |
| 9 | `vllm/v1/cudagraph_dispatcher.py:130` | `_create_padded_batch_descriptor()` | padding 后 `num_tokens` / `num_reqs` 怎么算 | LoRA case |
| 10 | `vllm/v1/cudagraph_dispatcher.py:324` | `get_capture_descs()` | 捕获哪些 bucket、为什么从大到小 | 3D / full graph |
| 11 | `vllm/v1/worker/gpu_worker.py:634` | `determine_available_memory()` | graph 显存也要从 KV 预算里扣 | 多模态 / IPC |
| 12 | `vllm/v1/worker/gpu_worker.py:908` | `compile_or_warm_up_model()` | 初始化顺序：warmup → kernel tune → capture | torch.compile 分支 |

### 2.2 建议读（选读）

| 位置 | 符号 | 为什么看 |
|---|---|---|
| `vllm/v1/worker/gpu_worker.py:962` | `capture_model()` 调用点 | 确认 capture 发生在 KV cache 分配之后 |
| `vllm/forward_context.py:30` | `BatchDescriptor` | vLLM 用 `num_tokens` / `num_reqs` / `uniform` 做 key；nano 只要一个 int |
| `vllm/v1/attention/backends/triton_attn.py:58` | `TritonAttentionMetadata` | 确认 attention 的每步变化都在 tensor 里，而不是 Python 标量 |

### 2.3 明确跳过

vLLM 的 piecewise/full graph、torch.compile、attention backend 的 graph 分支、DP/PP、
spec decode 的 graph、LoRA specialization、encoder graph。nano 只做“固定 batch bucket 的
decode graph”，prefill 一律 eager。

---

## 3. 核心概念

### 3.1 为什么 decode 适合，prefill 不适合

| 阶段 | 本步 token 数 | shape | nano 策略 |
|---|---|---|---|
| prefill | `prompt_len` 或 chunk size | 每步变化 | eager |
| decode | `batch_size`（每请求 1） | 只随 batch 变 | graph，按 bucket |

graph 不会让矩阵乘更快；它省的是 **launch 和 Python 开销**。所以 batch 越小、模型层数越多、
decode 越频繁，收益越明显。prompt 长度变化大，捕获所有 shape 不现实，收益也不稳定，nano
直接不做。

### 3.2 静态 buffer 清单

graph 捕获时，kernel 参数里记录的是这些 buffer 的**地址**。每个 bucket 一套：

| buffer | shape | dtype | 谁写 | 谁读 |
|---|---|---|---|---|
| `input_ids` | `[bucket]` | int64 | replay 前 H2D copy | embedding |
| `positions` | `[bucket]` | int64 | 同上 | RoPE |
| `slot_mapping` | `[bucket]` | int64 | 同上 | KV 写入 kernel |
| `seq_lens` | `[bucket]` | int32 | 同上 | attention kernel |
| `query_start_loc` | `[bucket+1]` | int32 | capture 时填死 `[0..bucket]` | attention kernel |
| `block_table` | `[bucket, max_blocks]` | int32 | replay 前 H2D copy | attention kernel |
| `logits` | `[bucket, vocab]` | bf16 | graph 内的 lm_head | `sample_tokens()` 读前 B 行 |
| `kv_caches` | 每层 `(k,v)` | bf16 | graph 内写/读 | attention |

注意：

- `query_start_loc` 对纯 decode 是常量 `[0,1,...,bucket]`，捕获时填一次即可；
- `seq_lens` 的真实值每步变，必须每步 copy；
- `slot_mapping` 在 vLLM 里属于 `BlockTable` 的 buffer，nano 把它提到 graph runner 里；
- **`num_reqs` / `max_query_len` / `causal` 是 Python 值，捕获时会被烧进 graph**。decode
  路径固定 `num_reqs=bucket`、`max_query_len=1`、`causal=True`，所以可以接受。

### 3.3 地址语义：什么能变，什么不能变

```text
capture 时记录：kernel 指针、tensor 地址、shape、Python 标量
replay 时读取：这些地址上的“当前内容”
```

因此：

- 可以变：`seq_lens` / `slot_mapping` / `block_table` / `input_ids` 的**元素值**；
- 不能变：tensor 的 shape、dtype、device、data_ptr；`bucket`；`num_reqs`；任何在 capture
  时参与 Python `if` / `range` 的值；
- 不能在 capture 后重新分配 KV cache 或模型权重：地址变了，graph 还在指旧地址。

这也是为什么 replay 前必须把新输入**拷进同一组 buffer**，而不是 new 一个 tensor。

### 3.4 bucket 与 dummy 行

bucket 取 `[1, 2, 4, 8, 16, 32]`。`batch_size=3` 用 4 号 bucket，第 4 行是 dummy：

```text
真实行: 0 1 2      seq_lens = [17, 5, 9]
dummy:  3          seq_lens = 1
block_table[3] = [pad_block, 0, ...]
slot_mapping[3] = pad_block * block_size + 0
output[3] 丢弃
```

dummy 行必须满足：

1. `seq_lens=1`，attention 只读一个 KV token；
2. `block_table` 指向**专用 pad block**，不能指向 0 号块（0 是真实块）；
3. `slot_mapping` 写 pad block，不能写真实请求的 slot；
4. 输出行丢弃，不参与采样。

graph 内把 bucket 行都当 actual：`num_reqs=bucket`、`num_actual_tokens=bucket`；`B` 只用于
输出截断。eager 路径仍然 `num_actual_tokens=B`。

pad block 在 `GPUWorker.initialize_kv_cache()` 时从 KV cache 末尾预留
`ceil(max_bucket / block_size)` 块，`BlockPool` 只使用前面的块。dummy 行按
`pad_block_start + i // block_size` 分行，避免多个 dummy 写同一个 slot。

### 3.5 graph 只接 Triton backend

naive backend 在 Python 里 `int(metadata.seq_lens[r])`，这个值在 capture 时会被烧成 dummy
的 `seq_len=1`，replay 时不会重新读。**所以 naive 不能进 graph**，它的定位始终是 eager
oracle。

Triton backend 的 `seq_lens` / `slot_mapping` / `block_table` 全部由 kernel 从 tensor 读，
Python 侧只读 `num_reqs`（固定 bucket）和 `max_query_len`（固定 1）。这是 M10 能成立的前提。
如果以后要让 naive 也 graph-safe，必须把所有 `.item()` 改成 GPU 端计算；本模块不做。

### 3.6 捕获协议

```text
1. 分配 KV cache、pad block；确定 buckets
2. 用 dummy 输入在 side stream 上 warmup 3 次（触发 lazy init / cuBLAS / Triton autotune）
3. torch.cuda.synchronize(); torch.cuda.empty_cache()
4. pool = torch.cuda.graph_pool_handle()
5. for bucket in sorted(buckets, reverse=True):
       g = torch.cuda.CUDAGraph()
       with torch.cuda.graph(g, pool=pool):
           out = model_forward(static_buffers[bucket])
       graphs[bucket] = (g, out)
6. torch.cuda.synchronize(); 记录 capture 前后 free memory
```

要点：

- **先大后小**：大 bucket 的 graph pool 可以覆盖小 bucket，vLLM 的 `capture_model()` 也这么做；
- warmup 不够时，第一次 replay 可能触发 allocator 重新分配或 Triton 编译，直接报错；
- capture 期间不能有 `.item()`、`.tolist()`、`torch.nonzero`、依赖输入长度的 `torch.arange`；
- capture 用共享 pool，避免每个 bucket 各占一份显存；
- `determine_available_memory()` 先扣 `graph_reserve_bytes` 再算 `num_gpu_blocks`；capture 后测
  实际 graph 显存，若超过 reserve，降 `num_gpu_blocks` 重来，不要硬扛。`graph_reserve_bytes`
  先给保守常量（如 256 MiB），跑通后用 capture 实测校准。

### 3.7 采样不进 graph

graph 只包 **model forward**。`sample_tokens()` 仍然是普通 PyTorch：temperature / top-k /
top-p / 随机数生成器 / stop 条件都在 graph 外。原因：

- 采样依赖 Python 逻辑和 RNG 状态，graph 捕获会把随机数固定；
- 采样输出是变长 list，不是固定 shape；
- 采样必须在 `execute_model()` 返回 `None` 之后由 `sample_tokens()` 做，保持 §6.6 的切分。

### 3.8 指标定义

```text
TTFT = 第一个 output token 的时间 - 请求开始时间
TPOT = (最后一个 token 时间 - 第一个 token 时间) / (num_output_tokens - 1)
throughput = 总 output tokens / 总 wall time
```

测量要求：warmup 3–5 轮；固定 prompt 集和固定 output 长度；用 `torch.cuda.Event` 或
`torch.cuda.synchronize()`；报多次运行的中位数。graph 只影响 TPOT，TTFT 仍由 prefill 决定。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 保留 / 删除 / 简化 |
|---|---|---|
| `gpu_model_runner.py:6482 capture_model()` | `DecodeGraphRunner.capture()` | 保留“按 bucket 捕获、大 bucket 先、capture 后测显存” |
| `cudagraph_dispatcher.py:15 CudagraphDispatcher` | 内部 `_bucket_for(batch_size)` | 删 piecewise/full/LoRA/DP key，只留 batch bucket |
| `forward_context.py:30 BatchDescriptor` | `int bucket` | 删 `num_reqs` / `uniform` / `has_lora` |
| `gpu_model_runner.py:5508 _dummy_run()` | `DecodeGraphRunner.warmup()` | 保留 dummy forward；删 LoRA/多模态 |
| `gpu_model_runner.py:6133 profile_run()` | `GPUWorker` 的 profiling | 保留峰值显存；graph 显存另算 |
| `gpu_worker.py:634 determine_available_memory()` | 同左 | 加 `graph_reserve_bytes` |
| full / piecewise graph | 无 | 只做 decode-only；prefill 永远 eager |
| `torch.compile` | 无 | nano 不编译 |

**本模块新增的内部接口**（不是 §6 公共契约；`ModelRunner` 的公开方法不变）：

```python
# nanovllm/worker/cuda_graph.py
@dataclass
class DecodeGraphBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    logits: torch.Tensor

class DecodeGraphRunner:
    def __init__(self, model, kv_caches, max_bucket, max_blocks,
                 pad_block_start, device): ...
    def capture(self, buckets: list[int]) -> None: ...
    def replay(self, batch_size: int, input_ids, positions,
               slot_mapping, seq_lens, block_table) -> torch.Tensor: ...
```

`ModelRunner.execute_model()` / `sample_tokens()` 的签名保持 §6.6；graph 只是
`execute_model()` 内部的一条分支。

---

## 5. 实现任务

> 按顺序做；每个任务先写失败测试。先做 batch=1 的 graph，再扩 bucket。

### T1 · 静态 buffer 设计（40 分钟）

文件：`nanovllm/worker/cuda_graph.py`

做：定义 `DecodeGraphBuffers`；所有 buffer 在构造时一次分配；H2D 用 pinned staging tensor；
`query_start_loc` 直接填 `[0..bucket]`；dtype 和 M06 对齐（不要中途换 dtype）。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "buffers"`

坑：capture 时临时 `torch.empty`；buffer 放在 CPU；`block_table` 宽度用了当前 batch 的块数
而不是 `max_blocks`。

### T2 · bucket + dummy 行（40 分钟）

文件：`cuda_graph.py`、`nanovllm/worker/gpu_worker.py`

做：buckets `[1,2,4,8,16,32]`，`bisect_left` 选最近 bucket；预留 pad blocks；dummy 行填
`seq_lens=1`、block table 指向 pad、slot 指向 pad；真实行 copy 后其余行清零。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "bucket or padding"`

坑：dummy 指向 0 号块；pad block 被 `BlockPool` 分配；`bucket=3` 时第 3 行没初始化，读到上
一步的脏 slot。

### T3 · 捕获 decode graph（90 分钟）

文件：`cuda_graph.py`、`nanovllm/worker/model_runner.py`

做：按 §3.6 的顺序 warmup / sync / 共享 pool / 大 bucket 先捕获；保存 `(graph, static_out)`；
capture 前后记录 free memory；失败时打印 bucket 和 traceback。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "capture"`

坑：warmup 不足，首次 replay 报 allocator 错；capture 中读 `.item()`；capture 后重新分配
KV cache 或模型权重。

### T4 · replay 路径（60 分钟）

文件：`model_runner.py`

做：`execute_model()` 判断“纯 decode + 有 bucket”时，把 5 个输入拷进静态 buffer，调
`g.replay()`，把 `logits[:B]` 存进 `execute_model_state`，返回 `None`；`sample_tokens()` 不变。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "replay"`

坑：只拷 `input_ids`，忘了 `positions` / `slot_mapping` / `seq_lens` / `block_table`；把采样
放进 graph；replay 后没同步就读 logits（同一 stream 其实有序，但跨 stream 会错）。

### T5 · prefill / chunked fallback（40 分钟）

文件：`model_runner.py`

做：`max_query_len > 1`、`total_num_tokens != num_reqs`（混合 batch）、`B > max_bucket`、
backend 不是 Triton、`graph_enabled=False` 时全部走 eager；日志里计数 fallback 原因。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "fallback"`

坑：只用 `total_num_tokens == num_reqs` 判断 decode，忘了 `max_query_len==1`；chunked 被误判
成 decode；prefill 也走了 graph。

### T6 · graph on/off 对拍（60 分钟）

文件：`tests/test_m10_cuda_graph.py`

做：同一组权重、同一 seed、greedy；batch=1/3/8 各跑 10 个 decode step，每步比 logits
（`atol=1e-3, rtol=1e-3`）和 token（必须完全一致）；prefill/chunked 的 logits 也要一致。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "parity"`

坑：graph 和 eager 用了不同 backend；比较了 padded 行；不同随机种子；只比最终文本，不比
每步 logits。

### T7 · pinned memory（30 分钟）

文件：`cuda_graph.py`

做：为 5 个输入各配一个 `pin_memory=True` 的 CPU staging tensor；`copy_(..., non_blocking=True)`
到 GPU 静态 buffer；benchmark 比较 pinned vs pageable 的 TPOT。

验收：`python -m pytest tests/test_m10_cuda_graph.py -q -k "pinned"`

坑：pinned tensor 每次 replay 重新分配（应该复用）；copy 没在 replay 前完成；小 tensor 上
pinned 收益不明显，别把噪声当加速。

### T8 · benchmark + 报告（40 分钟）

文件：`benchmarks/bench_m10_decode.py`

做：batch=1/8/32，graph on/off；固定 prompt 集、固定 output 长度；warmup 3 轮，正式跑 3 次
取中位数；输出 TTFT / TPOT / throughput 和加速比表格。

验收：`python benchmarks/bench_m10_decode.py --batch-size 1 8 32 --graph both`

坑：`time.time()` 不同步 GPU；把 warmup 算进结果；graph on/off 的输出长度不同。

---

## 6. 测试与对拍

### 6.1 必测 case

| case | 配置 | 断言 |
|---|---|---|
| bucket 命中 | B=1/3/8 | 分别命中 1/4/8；不 fallback |
| 10 步 decode | B=3，greedy | 每步 logits `allclose`，token 完全一致 |
| prefill | prompt=20 | 走 eager；logits 与 graph off 一致 |
| chunked | budget=5 | 走 eager；logits 与 graph off 一致 |
| 混合 batch | prefill + decode | 整体 fallback，不 crash |
| 显存 | 连续 50 步 | `memory_allocated` 不持续增长 |
| 性能 | B=1/8/32 | 表格见 §6.2 |

### 6.2 benchmark 表模板

```text
batch | TTFT(ms) | TPOT eager(ms) | TPOT graph(ms) | speedup | tok/s eager | tok/s graph
1     |          |                 |                |         |             |
8     |          |                 |                |         |             |
32    |          |                 |                |         |             |
```

TTFT 不应因 graph 明显变化（prefill 仍 eager）；TPOT 才是 graph 的收益点。如果 graph 更慢，
先检查是不是每步都在做 H2D copy 或重新构建 metadata。

### 6.3 常见失败现象

| 现象 | 最可能的原因 |
|---|---|
| 第一次 replay 报 allocator / capture 错 | warmup 不够，或 capture 中新建了 tensor |
| 输出和 eager 完全不同 | 只拷了 input_ids，`seq_lens` / block table 还是 dummy |
| 只有第 3 行（dummy）NaN | dummy 的 `seq_lens=0` 或 block table 指向 0 |
| B=1 对，B=3 错 | bucket padding 的 slot/block table 没填对 |
| graph 没有加速 | 每步重新分配 pinned buffer / 重新构建 metadata |
| 显存持续涨 | 每个 step new graph / new pool，或采样 buffer 泄漏 |

---

## 7. 常见坑

1. **捕获时用了动态 shape。** `torch.arange(num_tokens)` 依赖输入长度，replay 会错；bucket
   内所有 shape 必须常量。
2. **只拷贝 `input_ids`。** `positions` / `slot_mapping` / `seq_lens` / `block_table` 任何
   一个忘了拷，输出都会错，而且往往只在多步后暴露。
3. **采样进了 graph。** RNG 被固定，temperature/top-p 失效；采样必须在 graph 外。
4. **dummy 行影响真实行。** dummy 指向 0 号块、或写真实 slot，会污染别的请求；必须用
   dedicated pad block。
5. **warmup 不足。** 首次 replay 触发重新捕获 / allocator 报错；至少 3 次 dummy forward。
6. **用 `time.time()` 计时。** GPU 异步，必须 `torch.cuda.synchronize()` 或用 CUDA Event。
7. **把 prefill 也塞进 graph。** prompt 长度组合爆炸；nano 只 graph decode。
8. **capture 后重新分配 KV cache。** graph 里的地址失效；KV 必须在 capture 前分配好。
9. **naive backend 进 graph。** 它用 Python 读 `seq_lens`，replay 时永远是 capture 时的值。
10. **block table 的列宽不一致。** 静态 buffer 用 `max_blocks`，capture 和 replay 必须同宽；
    不要把 `[B, 实际块数]` 的 tensor 直接塞进去。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m10_cuda_graph.py -q` 全绿；
- [ ] batch=1/3/8 都能命中 bucket，10 步 decode 的 logits 和 token 与 eager 一致；
- [ ] prefill / chunked / 混合 batch 全部 fallback，结果与 eager 一致；
- [ ] 连续 50 步 decode 后 `memory_allocated` 不持续增长；
- [ ] `benchmarks/bench_m10_decode.py` 输出 §6.2 的表；
- [ ] 能解释为什么 naive 不能进 graph、为什么采样不能进 graph；
- [ ] `PROGRESS.md` 已更新，commit：`M10: decode cuda graph + benchmark`。

---

## 9. 交接：给 M11 / M12

**新增/修改文件**

```text
nanovllm/worker/cuda_graph.py       # DecodeGraphRunner（内部）
nanovllm/worker/model_runner.py     # execute_model 的 graph 分支
nanovllm/worker/gpu_worker.py       # pad block / graph 显存 reserve
tests/test_m10_cuda_graph.py
benchmarks/bench_m10_decode.py
```

**公开 API**：§6.6 的 `ModelRunner.execute_model` / `sample_tokens` 签名不变；
`DecodeGraphRunner` 是模块内部类，不导出。

**已知限制**

- 只支持纯 decode、Triton backend、`max_query_len=1`、`B <= 32`；
- bucket 固定为 1/2/4/8/16/32；更大的 batch 直接 fallback；
- 不做 piecewise / full graph、torch.compile、spec decode graph、LoRA specialization；
- graph 显存是静态占用，必须在 `determine_available_memory()` 里预留。

**给 M12 的输入**

- `benchmarks/bench_m10_decode.py --graph both` 的表格直接进 `REPORT.md`；
- 报告里写清：TTFT 基本不变、TPOT 的加速比、哪些 batch 没收益、graph 显存开销。

---

## 10. 自测题

1. 为什么 CUDA graph 要求静态地址？replay 前要做什么？
2. 为什么 decode 适合 graph，prefill 不适合？
3. bucket padding 后，dummy 行怎么保证不影响真实输出？
4. 为什么采样不能放进 graph？
5. TTFT 和 TPOT 分别怎么测？为什么要 warmup？

---

## 11. 教练流程（给 agent 用）

20 min 讲 §3.1–§3.3 的地址语义 → 30 min 带读 `capture_model` / `CudagraphDispatcher` →
30 min 设计静态 buffer → 90 min 捕获/replay → 40 min 对拍/debug → 30 min benchmark →
15 min 复盘。

禁止：一上来做 full graph；先扩 bucket 再跑通 batch=1；把采样放进 graph；用
`time.time()` 不同步；禁止让人通读 7200 行 runner。**先让 batch=1、单步 replay 和 eager 完全
一致，再扩到 3/8 和 10 步。**
