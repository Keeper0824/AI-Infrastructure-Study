# 文档作者 brief C：M04 Paged Attention + M10 CUDA Graph/性能

## 通用要求

工作区：`/home/dictator/ai_infra/nanovllm`

先读：

1. `docs/00_architecture.md`（§5–§8，尤其 §6.5 注意力契约）
2. `docs/00_facts.md`（环境、Qwen3、行号、写作约定）
3. `docs/M00_contracts.md`（风格）
4. `docs/_template.md`
5. `ROADMAP.md`
6. `docs/_briefs/B_kv.md` 的 M03 部分（了解 KV layout / BlockTable / slot mapping）

只写：

- `docs/M04_paged_attention.md`
- `docs/M10_cuda_graph.md`

不要改代码/其他文档。每个文件 350–500 行。中文。

---

## M04：Paged Attention：后端抽象 + Naive + Triton

### 目标

实现 `AttentionBackend` 抽象，两个后端：

1. `NaivePagedAttention`：把 block table 指向的 K/V gather 成连续 tensor，用 PyTorch SDPA
   或手写 softmax；慢，但正确，作为 oracle。
2. `TritonPagedAttention`：一个 Triton kernel，直接从分页 KV cache 读，写 output；快。

并把它们接进 Qwen3 的 attention 层，替换 M02 的连续 cache。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/attention/backend.py:59` | `AttentionBackend` | 抽象方法、`get_name` / `get_impl_cls` / `get_builder_cls` |
| `vllm/v1/attention/backend.py:379` | `AttentionMetadata` | 空基类；vLLM 用子类 |
| `vllm/v1/attention/backend.py:387` | `CommonAttentionMetadata` | `query_start_loc` / `seq_lens` / `block_table_tensor` / `slot_mapping` |
| `vllm/v1/attention/backend.py:589` | `AttentionMetadataBuilder` | 从 Common 构建 backend-specific metadata |
| `vllm/v1/attention/backends/triton_attn.py:58` | `TritonAttentionMetadata` | 字段和注释（context_len/query_len/seq_len） |
| `vllm/v1/attention/backends/triton_attn.py:209` | `TritonAttentionMetadataBuilder.build()` | 只做字段拷贝；nano 不需要 builder 继承树 |
| `vllm/v1/attention/backends/triton_attn.py:385` | `TritonAttentionImpl` | `forward()` 的调用链 |
| `vllm/v1/attention/backends/triton_attn.py:549` | `TritonAttentionImpl.forward()` | 选 2D/3D kernel；nano 只做一种 |
| `vllm/v1/attention/ops/triton_unified_attention.py:179` | `kernel_unified_attention` | 只看循环结构：block 循环、QK、softmax、PV；不要逐行读优化 |
| `vllm/v1/attention/ops/triton_unified_attention.py:815` | `unified_attention()` | grid、`num_queries_per_kv`、调用参数 |
| `vllm/v1/worker/block_table.py:153` | `compute_slot_mapping()` | slot 计算 |
| `vllm/v1/worker/gpu_model_runner.py:1920` | `_prepare_inputs` | `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping`、`logits_indices` |
| 可选 | `vllm/v1/attention/backends/flash_attn.py` | 对比另一个 backend 的 metadata；不必读实现 |

**明确跳过**：vLLM 的 2D/3D 调度、segment softmax、quant、alibi、sliding window、MLA、
CUDA graph 分支、Triton descriptor、FP8。nano 只做 BF16、causal、GQA、单 backend。

### 必须讲清的概念

1. **AttentionMetadata 的五个张量**：
   - `query_start_loc`: `[num_reqs+1]`，第 r 个请求的 query 在扁平 token 维的起止；
   - `seq_lens`: `[num_reqs]`，每个请求当前总长（context + query）；
   - `block_table`: `[num_reqs, max_blocks]`，逻辑块 → 物理块；
   - `slot_mapping`: `[num_tokens]`，本步每个 token 写到哪个物理 slot；
   - `causal`: prefill 为 True；decode 只有 1 个 query，mask 退化为无。
2. **KV cache layout**：`k_cache: [num_blocks, block_size, num_kv_heads, head_dim]`；
   `slot = block_id * block_size + offset`。
3. **GQA**：`num_q_heads=16`，`num_kv_heads=8`，`group_size=2`；
   `kv_head = q_head // group_size`。
4. **causal mask 的绝对位置**：query 的绝对位置 = `seq_len - query_len + local_q_idx`；
   key 的绝对位置 = `block_idx * block_size + offset`；mask 条件 `key_pos <= query_pos`。
   chunked prefill 时 query 不是从 0 开始，不能用 `local_q_idx` 当绝对位置。
5. **online softmax**（Triton）：维护 `m_i`（running max）、`l_i`（running sum）、
   `acc`（running output）；每读一个 block 更新。公式要写出来。
6. **为什么 naive 先做**：Triton 出错时，naive 是唯一能告诉你“是 kernel 错还是数据错”的 oracle。
7. **cache 写入**：`k_cache[slot] = k`；Triton 可以用 `tl.store`，也可以先用 PyTorch
   `index_copy`；M10 再优化。

### 必须实现的接口

严格按 `docs/00_architecture.md` §6.5：

```python
@dataclass
class AttentionMetadata:
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor   # int32, [num_reqs+1]
    seq_lens: torch.Tensor          # int32, [num_reqs]
    block_table: torch.Tensor       # int32, [num_reqs, max_blocks]
    slot_mapping: torch.Tensor      # int64, [num_tokens]
    causal: bool = True

class AttentionBackend(ABC):
    @abstractmethod
    def forward(self, q, k, v, kv_cache, metadata) -> torch.Tensor: ...
```

`q`: `[T, 16, 128]`；`k/v`: `[T, 8, 128]`；`kv_cache`: `(k_cache, v_cache)`，形状
`[num_blocks, block_size, 8, 128]`；返回 `[T, 16, 128]`。

`NaivePagedAttention.forward` 的步骤（文档里写清楚）：

```text
1. 把 k/v 写到 cache：k_cache.view(-1, 8, 128)[slot_mapping] = k
2. for each request r:
     seq_len = seq_lens[r]; q_start, q_end = query_start_loc[r:r+2]
     blocks = block_table[r, :ceil(seq_len/block_size)]
     k_all = k_cache[blocks].reshape(-1, 8, 128)[:seq_len]
     v_all = ...
     q_r = q[q_start:q_end]                  # [q_len, 16, 128]
     k_r = repeat_kv(k_all, 2)               # [seq_len, 16, 128]
     attn = sdpa(q_r, k_r, v_r, is_causal=(q_len>1))
3. 拼回 [T, 16, 128]
```

注意 `sdpa` 的 `is_causal` 在 q_len>1 且 q_start>0 时是“左上角对齐”的 mask，不是绝对位置；
chunked prefill 必须手写 mask，或用 `attn_mask` 显式构造。文档必须讲这个坑。

`TritonPagedAttention.forward`：

- grid = `(num_tokens, num_q_heads)`（先简单）；
- program_id(0)=token 在扁平 batch 的位置，program_id(1)=q head；
- 用 `query_start_loc` 二分/线性找 request；
- `q_pos = token_pos - q_start`；绝对位置 `pos = seq_len - q_len + q_pos`；
- 循环 `block_idx` 从 0 到 `ceil(seq_len/block_size)`；
- 读 K/V block，算 `q·k * scale`，mask `key_pos > pos`，online softmax；
- 写 output。

文档里给出伪代码（不是完整 kernel），并解释 `BLOCK_N`、`HEAD_DIM`、`num_warps` 的选择。

### 实现任务

T1 `AttentionMetadata` + backend ABC；T2 naive backend + KV 写入；T3 接进 Qwen3；
T4 三路对拍（M02 连续 / naive / Triton）；T5 Triton decode kernel；T6 Triton prefill/chunked；
T7 GQA + causal 测试；T8 benchmark（decode tokens/s、naive vs Triton）。

### 测试与对拍

- 固定随机 q/k/v 和 block table，naive vs Triton `assert_close(atol=1e-3, rtol=1e-3)`；
- 同一 prompt，M02 连续 cache vs M04 paged 的 logits 一致；
- prefill（q_len=prompt_len）、decode（q_len=1）、chunked（q_len=3，q_start=13）；
- GQA：16 个 q head 的输出不能全一样；检查第 0/1 个 head 共享同一个 kv head；
- 边界：seq_len=1、seq_len=block_size、seq_len=block_size+1、block table 有 padding；
- 性能：decode batch=1/8/32，Triton 应明显快于 naive。

### 常见坑

- `seq_lens` 是“总长”，不是“本步 query 长度”；
- `query_start_loc` 是前缀和，不是每个请求的长度；
- causal mask 用了 `local_q_idx`，chunked prefill 时错；
- `block_table` 的 padding 行指向 0 号块，读到了别的请求的数据；
- GQA 的 `kv_head = q_head // group_size` 写反；
- Triton 的 `tl.dot` 要求 shape 是 2 的幂；`head_dim=128` 没问题，`BLOCK_N` 选 16/32/64；
- `tl.load` 的 mask 没加，最后一个 block 越界；
- 输出 dtype 是 fp32，忘了 cast 回 bf16；
- cache 写入和 attention 读的 layout 不一致（`[B, H, D]` vs `[B, D, H]`）。

### 自测题

1. `query_start_loc` 和 `seq_lens` 各是什么？为什么两个都需要？
2. 为什么 decode 的 causal mask 可以省掉？
3. chunked prefill 时，怎么算 query 的绝对位置？
4. GQA 的 16 个 Q head 怎么映射到 8 个 KV head？
5. online softmax 的 `m_i` / `l_i` / `acc` 分别怎么更新？

### 教练流程

概念 20min → 读 backend/metadata 30min → 写 naive 60min → 三路对拍 30min →
写 Triton decode 90min → prefill/chunked 60min → benchmark 30min → 复盘 15min。
先让 naive 跑通，再写 Triton；不要一上来写 kernel。

---

## M10：CUDA Graph 与性能

### 目标

在 M06 的 decode 路径上做 **decode-only CUDA graph**：

- 预分配静态输入 buffer（input_ids / positions / slot_mapping / seq_lens / block_table）；
- 按 batch size bucket（1,2,4,8,16,32...）捕获 graph；
- decode 时把真实输入拷进静态 buffer，replay graph，取 logits；
- prefill / chunked prefill 继续走 eager；
- 加 pinned memory、benchmark（TTFT/TPOT/throughput）。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/worker/gpu_model_runner.py:6482` | `capture_model()` | 捕获哪些 shape、bucket 从哪来 |
| `vllm/v1/worker/gpu_model_runner.py:6133` | `profile_run()` | warmup / 显存 profiling |
| `vllm/v1/worker/gpu_model_runner.py:5508` | `_dummy_run()` | 构造 dummy batch |
| `vllm/v1/worker/gpu_model_runner.py:467` | `GPUModelRunner.__init__` 的 buffer 部分 | 静态 buffer 有哪些 |
| `vllm/v1/cudagraph_dispatcher.py` | `CUDAGraphDispatcher` | batch descriptor / shape 选择 |
| `vllm/v1/worker/gpu_worker.py:634` | `determine_available_memory()` | graph 显存也要算进预算 |
| `vllm/v1/worker/gpu_worker.py:908` | `compile_or_warm_up_model()` | 初始化顺序 |
| 可选 | `vllm/compilation/cuda_graph.py` | 更上层的封装；nano 不用 |

**明确跳过**：piecewise/full graph、torch.compile、attention backend 的 graph 分支、
DP/PP、spec decode 的 graph。nano 只做“固定 batch bucket 的 decode graph”。

### 必须讲清的概念

1. **为什么 decode 适合 graph**：每步 shape 几乎固定（batch、1 token/req），CPU launch
   开销占比高；graph 把几十个 kernel launch 合成一次 replay。
2. **为什么 prefill 不适合（nano）**：prompt 长度变化大，shape 不固定；先 eager。
3. **静态 buffer**：graph 捕获时记录的是**地址**；replay 前必须把新输入拷到同一地址。
4. **bucket**：把 batch size padding 到最近的 bucket（1/2/4/8/...），graph 只认 bucket；
   多出来的行用 dummy 请求/0 填充，输出丢弃。
5. **block table 也是输入**：每步可能变；必须拷贝到静态 buffer，不能传新 tensor。
6. **采样不要进 graph**：graph 只包 forward；采样在 graph 外用 PyTorch 做（随机数/动态）。
7. **graph 与 allocator**：捕获期间不能有动态分配；warmup 后 `torch.cuda.synchronize()`；
   用 `torch.cuda.graph(g)` 上下文。
8. **性能指标**：TTFT、TPOT、throughput；warmup、固定 prompt 集、固定 output 长度；
   同步计时（`torch.cuda.synchronize`）。

### 实现任务

T1 静态 buffer 设计；T2 batch bucket + padding；T3 捕获 decode graph；T4 replay 路径；
T5 prefill fallback；T6 graph on/off 对拍；T7 pinned memory；T8 benchmark 脚本 + 报告。

### 测试与对拍

- 同一 batch，graph on/off 的 logits `assert_close`；
- 不同 batch size（1/3/8）都能命中 bucket；
- 连续 decode 10 步，输出与 eager 完全一致；
- prefill/chunked 仍走 eager，结果一致；
- 显存：graph 捕获后 `torch.cuda.memory_allocated` 不持续增长；
- benchmark：batch=1/8/32，TTFT/TPOT/throughput 表格；graph 对 decode 的加速比。

### 常见坑

- 捕获时用了动态 shape（`torch.arange` 依赖输入长度），replay 直接错；
- 只拷贝 input_ids，忘了 positions/slot_mapping/seq_lens/block_table；
- 采样进了 graph，随机数固定/不同步；
- bucket padding 的 dummy 行影响了真实行的 attention（block table/seq_lens 没填对）；
- graph 捕获前的 warmup 不够，首次 replay 触发重新捕获/allocator 报错；
- 用 `time.time()` 计时，没同步 GPU；
- 把 prefill 也塞进 graph，shape 爆炸。

### 自测题

1. 为什么 CUDA graph 要求静态地址？replay 前要做什么？
2. 为什么 decode 适合 graph，prefill 不适合？
3. bucket padding 后，dummy 行怎么保证不影响真实输出？
4. 为什么采样不能放进 graph？
5. TTFT 和 TPOT 分别怎么测？为什么要 warmup？

### 教练流程

概念 20min → 读 capture_model/profile_run 30min → 设计静态 buffer 30min →
捕获/replay 90min → 对拍/debug 40min → benchmark 30min → 复盘 15min。
先做 batch=1 的 graph，再扩 bucket；不要一上来做 full graph。

---

## 两篇文档的衔接

- M04 依赖 M03 的 `BlockTable` / slot mapping；M04 的输出是 M06 的输入。
- M10 依赖 M06 的 `ModelRunner`；不要改 M04 的 kernel 接口。
- M04 的 naive backend 要保留到 M10/M12 作为对拍 oracle。
- M10 的 benchmark 结果留给 M12 的 REPORT。

## 交付前检查

- [ ] 两个文件 350–500 行；
- [ ] M04 有 naive 的逐步伪代码和 Triton 的 grid/循环说明；
- [ ] 明确 chunked prefill 的 causal mask 坑；
- [ ] M10 有静态 buffer 清单、bucket 策略、graph on/off 对拍；
- [ ] 两篇都有性能测试方法和常见坑；
- [ ] 没有建议读完整 7000 行 runner。
