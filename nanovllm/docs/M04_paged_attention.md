# M04 · Paged Attention：后端抽象 + Naive + Triton

> **模块目标**：把 M03 的分页 KV cache 真正接进 attention：先写一个 gather + SDPA 的
> naive 后端当 oracle，再写一个直接从分页 cache 读 K/V 的 Triton kernel。
> **预计工时**：4–5 小时　**依赖**：M02（连续 cache oracle）、M03（block table / slot mapping）　**测试**：GPU
>
> **产出**：`pytest tests/test_m04_*.py -q` 全绿；连续 / naive / Triton 三路 logits 对拍；
> `benchmarks/bench_m04_attention.py` 的 decode tokens/s 表。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m04_backend.py tests/test_m04_naive.py -q
python -m pytest tests/test_m04_triton.py -q          # 需要 GPU
python -m pytest tests/test_m04_model_parity.py -q
python benchmarks/bench_m04_attention.py --batch-size 1 8 32
```

以及能不看代码回答 §10 的五个问题。

---

## 1. 为什么需要这个模块（10 分钟）

M02 的 attention 直接读写一份连续 KV cache；M03 把 cache 切成了物理块，请求只持有
block table，本步 token 通过 slot mapping 写入。**attention 不能再假设 K/V 连续**：
它必须拿着 `block_table` 去物理块里 gather，或者让 kernel 自己按块读。

本模块只做三件事：定义 `AttentionMetadata` / `AttentionBackend`；写
`NaivePagedAttention`（gather + SDPA，慢但可打印）；写 `TritonPagedAttention`
（一个 kernel 直接读分页 cache）。

边界：**M03** 负责分配/释放 block、维护 `BlockTable`、算 `slot_mapping`；**M04** 消费
`(k_cache, v_cache)` 和 metadata，完成“写 cache + attention”；**M06** 每步填 metadata，
本模块测试用手工构造的 fake metadata。

本模块不改 `docs/00_architecture.md` §6.5 的字段；模型层内部 forward 签名会变，但那是
模块内部约定，不是跨模块公共接口（见 §4）。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`
> 引用格式：`vllm/v1/...py:行号 · 符号`；行号漂了用 `grep -n "<符号>"` 重定位。
> `backend.py` 有 1100+ 行、`gpu_model_runner.py` 有 7200+ 行，**不要通读**；只读下表
> 锚点前后 20–60 行。

### 2.1 必读（按顺序）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/attention/backend.py:59` | `AttentionBackend` | 三个抽象方法 `get_name` / `get_impl_cls` / `get_builder_cls`；`forward_includes_kv_cache_update` 表示 forward 里含 cache 写入 | 平台支持矩阵、MLA/量化判断 |
| 2 | `vllm/v1/attention/backend.py:379` | `AttentionMetadata` | 空基类；vLLM 让每个 backend 定义自己的子类 | 这个类本身只有 `pass`，不要找字段 |
| 3 | `vllm/v1/attention/backend.py:387` | `CommonAttentionMetadata` | 五个张量 + `num_actual_tokens` / `max_query_len`；为什么有 CPU/GPU 两份 | LoRA / 多模态 / spec 字段 |
| 4 | `vllm/v1/attention/backend.py:589` | `AttentionMetadataBuilder` | “从 Common 构建 backend 私有 metadata”这个职责；nano 不需要继承树 | 继承层次、reorder、CUDA graph 支持 |
| 5 | `vllm/v1/attention/backends/triton_attn.py:58` | `TritonAttentionMetadata` | 字段：`query_start_loc` / `seq_lens` / `block_table` / `slot_mapping` / `causal`；context/query/seq 的注释 | 3D softmax、quant、多模态、rswa |
| 6 | `vllm/v1/attention/backends/triton_attn.py:209` | `TritonAttentionMetadataBuilder.build()` | 只做字段拷贝：Common → Triton 私有；nano 不需要 builder | 后面的 mm/rswa 分支、`build_for_cudagraph_capture` |
| 7 | `vllm/v1/attention/backends/triton_attn.py:385` | `TritonAttentionImpl` | 构造函数里的 `num_heads_q` / `num_heads_kv` / `headdim`；`forward()` 的调用链 | 量化、descriptor、alibi |
| 8 | `vllm/v1/attention/backends/triton_attn.py:549` | `TritonAttentionImpl.forward()` | 取 metadata → 处理 cache view → 调 `unified_attention`；真正的 2D/3D 分支在 launcher，forward 只传 `seq_threshold_3D` | encoder 分支、量化分支 |
| 9 | `vllm/v1/attention/backends/triton_attn.py:665` | `unified_attention(...)` 调用点 | 实际传了哪些参数；哪些是 nano 不需要的 | 后面所有 alibi/quant/sink 参数 |
| 10 | `vllm/v1/attention/ops/triton_unified_attention.py:179` | `kernel_unified_attention` | **只看循环骨架**：按 KV block 循环 → QK → mask → online softmax → PV | 2D/3D 分支、segment softmax、descriptor、FP8 |
| 11 | `vllm/v1/attention/ops/triton_unified_attention.py:815` | `unified_attention()` | grid、`num_queries_per_kv`、调用参数 | tile size 调优、batch invariant |
| 12 | `vllm/v1/attention/ops/triton_unified_attention.py:979` | `total_num_q_blocks` | 2D grid 的第一维怎么估出来 | 3D 的 `num_par_softmax_segments` |
| 13 | `vllm/v1/attention/ops/triton_unified_attention.py:1091` | `grid = ...` | 2D 是 `(num_q_blocks, num_kv_heads)`，3D 多一个 softmax 段维度 | 3D 分支 |
| 14 | `vllm/v1/worker/block_table.py:153` | `compute_slot_mapping()` | vLLM 把 `block_id * block_size + offset` 放进 Triton kernel；nano 用 PyTorch 算同样的公式 | DCP/CP interleave、Mamba 分支 |
| 15 | `vllm/v1/worker/gpu_model_runner.py:1920` | `_prepare_inputs()` | `query_start_loc` 在 `:2029`、`positions` 在 `:2135`、`seq_lens` 在 `:2139`、slot mapping 在 `:2144` | spec / mrope / async scheduling |

### 2.2 建议读（选读）

| 位置 | 符号 | 为什么看 |
|---|---|---|
| `vllm/v1/attention/backends/flash_attn.py:513` | `FlashAttentionMetadata` | 对比另一个 backend 的 metadata 字段；不必读实现 |
| `vllm/v1/attention/backends/triton_attn.py:568` | `forward()` docstring | vLLM 的 Triton cache 布局是 `[num_blocks, num_kv_heads, block_size, 2*head_dim]`，和 nano 不同，别照抄 |
| `vllm/v1/worker/gpu_model_runner.py:2139` | `seq_lens = num_computed + num_scheduled` | 确认 `seq_lens` 是“总长”，不是本步 query 长度 |

### 2.3 明确跳过

vLLM 的 2D/3D 调度、segment softmax、quant（FP8/INT8/INT4）、alibi、sliding window、
MLA、CUDA graph 分支、Triton descriptor、CP/DCP。nano 只做 **BF16、causal、GQA、
单 backend、固定 `block_size=16`**。

---

## 3. 核心概念

### 3.1 metadata 的五个张量
设 `block_size=16`，batch 里有 A、B 两个请求：

- A 已算 17 个 token，本步 chunk 3 个；总长 20；
- B 已算 4 个 token，本步 decode 1 个；总长 5。

```text
num_reqs            = 2
num_actual_tokens   = 4                 # 本步真实 token 数，不是 padding 后
max_query_len       = 3                 # A 的 chunk 长度
query_start_loc     = [0, 3, 4]         # 前缀和；第 r 个请求是 [loc[r], loc[r+1])
seq_lens            = [20, 5]           # 每请求的“总长” = context + query
block_table         = [[b0, b1],        # [num_reqs, max_blocks]
                       [b2,  0]]        # B 只需 1 块，第二列是 padding
slot_mapping        = [s0, s1, s2, s3]  # 本步 4 个 token 写到哪个物理 slot
causal              = True
```

- `query_start_loc` 是**扁平 batch 偏移**，不是序列位置；A 的 `q_start=0` 不代表绝对位置 0。
- `seq_lens` 是**总长**；本步 query 长度是 `query_start_loc[r+1] - query_start_loc[r]`。
- `block_table` 宽度是 `max_blocks`，只有前 `ceil(seq_len/block_size)` 列有效。
- `slot_mapping` 长度是 `num_actual_tokens`，不是 `num_reqs`。

### 3.2 KV cache layout 与 slot

nano 只认一种布局（§6.5 冻结）：

```text
k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
         [N,          16,         8,            128]
v_cache: 同形状
slot    = block_id * block_size + offset
```

所以整块 cache 可以 view 成 `[num_blocks * block_size, 8, 128]`，slot 就是这个二维表里的
行号。写入和读取必须用同一个 view；`[B, H, D]` 与 `[B, D, H]` 混用会静默算错。
vLLM 的 Triton 后端用 `[num_blocks, num_kv_heads, block_size, 2*head_dim]`（见
`triton_attn.py:568` 的 docstring），**不要带进 nano**。

### 3.3 GQA：16 个 Q head 共享 8 个 KV head

Qwen3-0.6B：`num_q_heads=16`，`num_kv_heads=8`，`group_size = 16 // 8 = 2`。

```text
kv_head = q_head // 2
q_head:  0  1  2  3  4  5 ... 14 15
kv_head: 0  0  1  1  2  2 ...  7  7
```

naive 扩 KV 必须用 `k.repeat_interleave(2, dim=1)`，不能用 `k.repeat(1, 2, 1)`：后者得到
`[kv0..kv7, kv0..kv7]`，head 1 读到 kv1（应为 kv0）、head 2 读到 kv2（应为 kv1）。
Triton 里不用 repeat，直接 `kvh = qh // 2` 读对应 KV head。

### 3.4 causal mask 用的是绝对位置

```text
query 绝对位置 = seq_len - query_len + local_q_idx
key   绝对位置 = block_idx * block_size + offset
可见条件       = key_pos <= query_pos
```

例子：prompt 20，`block_size=16`，本步算第 16–19 个 token（`q_len=4`）：query 绝对位置是
16/17/18/19，第一个 query 能看 key 0..16，最后一个能看 key 0..19。

**坑**：`F.scaled_dot_product_attention(..., is_causal=True)` 的 mask 是“左上角对齐”的
`mask[i, j] = j <= i`。当 `q_len == seq_len` 时它等价于绝对 causal；当 `q_len < seq_len`
时它会把“q 的第 0 行”当成“序列的第 0 个 token”，chunked prefill 直接错。只有
`q_start_in_seq == 0 && q_len == seq_len`（整段 prefill）或 `q_len == 1`（decode）时能安全
用 `is_causal=True`；其他情况必须显式构造 `[q_len, seq_len]` bool mask。注意
`query_start_loc[r]` 是扁平 batch 偏移，序列内起点是 `seq_len - q_len`。

### 3.5 online softmax

Triton 每读一个 KV block 更新一次；维护 `m_i`（running max）、`l_i`（running sum）、`acc`（fp32）：

```text
m_new = max(m_i, max_j s_j)
p_j   = exp(s_j - m_new)                 # 被 mask 的 s_j = -inf，p_j = 0
l_new = l_i * exp(m_i - m_new) + sum_j p_j
acc   = acc * exp(m_i - m_new) + sum_j p_j * v_j
m_i   = m_new
```

初始 `m_i=-inf`、`l_i=0`、`acc=0`；循环结束 `out=acc/l_i`。中间量全用 fp32，最后 cast 回
bf16。不要把 `m_i` 初始化成 0：第一个 block 的 scores 可能全负。

### 3.6 为什么先写 naive

对拍链条：`HF eager → M02 连续 cache → naive paged → Triton paged`。Triton 出错时通常只有
“乱码/NaN/某些 head 不对”，无法判断是 kernel、mask 还是 metadata；naive 每一步都可打印。
**naive 不是脚手架，M10/M12 还要拿它当 oracle，必须保留。**

### 3.7 naive 的步骤

```text
reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)
for r in range(num_reqs):
    qs, qe = int(query_start_loc[r]), int(query_start_loc[r+1])
    S, qlen = int(seq_lens[r]), qe - qs
    nblk = ceil(S / block_size)
    k_all = k_cache[block_table[r, :nblk].long()].reshape(-1, H_kv, D)[:S]
    v_all = v_cache[block_table[r, :nblk].long()].reshape(-1, H_kv, D)[:S]
    q_r = q[qs:qe].transpose(0, 1)                       # [Hq, qlen, D]
    k_r = k_all.repeat_interleave(2, dim=1).transpose(0, 1)
    v_r = v_all.repeat_interleave(2, dim=1).transpose(0, 1)
    if qlen == 1:
        out_r = sdpa(q_r, k_r, v_r)                      # decode：无 mask
    elif S == qlen:
        out_r = sdpa(q_r, k_r, v_r, is_causal=True)      # 整段 prefill
    else:
        q_pos = S - qlen + arange(qlen)                  # 绝对位置
        mask = arange(S)[None, :] <= q_pos[:, None]
        out_r = sdpa(q_r, k_r, v_r, attn_mask=mask)      # chunked
    out[qs:qe] = out_r.transpose(0, 1)
```

`reshape_and_cache` 就是 `k_cache.view(-1, H_kv, D).index_copy_(0, slot_mapping, k)`（v 同理）。
不要用 `k_cache[slot_mapping] = k` 直接写四维 tensor（索引维度对不上）。

### 3.8 Triton kernel 的骨架

grid 先用最简单的 `(num_actual_tokens, num_q_heads)`：一个 program 负责一个 token 的一个
Q head。这样 prefill 和 decode 共用同一个 kernel，只是 `qlen` 不同。

```text
grid = (T, 16);  tok = program_id(0);  qh = program_id(1)
req = searchsorted(query_start_loc, tok, right=True) - 1   # host 预算或 kernel 内二分
qs = query_start_loc[req]; S = seq_lens[req]; qlen = query_start_loc[req+1] - qs
pos = S - qlen + (tok - qs);  kvh = qh // 2               # 绝对位置
q = load(q_ptr + tok*stride_q_t + qh*stride_q_h + arange(D))
m = -inf; l = 0; acc = zeros(D, fp32)
for n0 in range(0, S, BLOCK_N):
    offs = n0 + arange(BLOCK_N);  mask_n = offs < S
    blk = load(block_table_ptr + req*bt_stride + offs // BLOCK_SIZE)
    slot = blk * BLOCK_SIZE + offs % BLOCK_SIZE
    k = load(k_cache_ptr + slot[:,None]*stride_k_slot
             + kvh*stride_k_head + arange(D)[None,:], mask=mask_n[:,None])
    v = load(...同样...)
    s = sum(q[None,:] * k, axis=1) * scale
    s = where(mask_n & (offs <= pos), s, -inf)
    m_new = maximum(m, max(s));  alpha = exp(m - m_new)
    p = exp(s - m_new)
    l = l * alpha + sum(p)
    acc = acc * alpha + sum(p[:,None] * v, axis=0)
    m = m_new
out[tok, qh, :] = (acc / l).to(bf16)
```

参数选择：`BLOCK_SIZE` 是 **KV page 大小**（nano 固定 16），由 M03 决定；`BLOCK_N` 是每次
循环读多少 KV token，decode 选 32、prefill 可选 64；`HEAD_DIM=128`，`num_warps=4` 起步。
单 token 版本没有 `tl.dot`；若改成多 query tile，`[BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N]`
每一维都必须是 2 的幂，`head_dim=128` 正好满足。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 保留 / 删除 / 简化 |
|---|---|---|
| `backend.py:59 AttentionBackend` | `attention/backend.py` | 只保留一个 `forward`；删 `get_builder_cls` / 平台矩阵 |
| `backend.py:387 CommonAttentionMetadata` | `AttentionMetadata` | 只保留五个张量 + `causal`；删 CPU/GPU 双份、max_seq_len |
| `backend.py:589 AttentionMetadataBuilder` | 无 | nano 在 M06 的 `_prepare_inputs` 里直接填 dataclass |
| `triton_attn.py:385 TritonAttentionImpl` | `backends/triton.py` | 只保留 BF16、causal、GQA |
| `triton_unified_attention.py:179 kernel` | `ops/paged_attention.py` | 只保留“一 token 一 head 一 program”的 2D 版本 |
| `triton_attn.py:665 unified_attention` | `TritonPagedAttention.forward` | 不做 2D/3D 阈值、不做 segment softmax |
| `block_table.py:153 compute_slot_mapping` | M03 的 `BlockTable.compute_slot_mapping` | M04 只消费结果 |
| `gpu_model_runner.py:1920 _prepare_inputs` | M06 的 `_prepare_inputs` | M04 只定义 metadata 字段 |
| vLLM cache layout `[N,H,B,2D]` | `[N,B,H,D]` | 定死 nano 布局，两套 cache 不能混用 |

**冻结接口**（严格按 `docs/00_architecture.md` §6.5，不新增字段）：

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

两个后端类名固定为 `NaivePagedAttention` / `TritonPagedAttention`，都实现同一个
`forward`。`forward` **同时负责写 cache 和做 attention**，模型层不碰 slot mapping。

**模型内部签名（本模块修改，不属于 §6 公共契约）**：

```python
# nanovllm/models/qwen3.py
Qwen3Attention.forward(hidden_states, positions, kv_cache, attn_metadata)
Qwen3Model.forward(input_ids, positions, kv_caches, attn_metadata)
```

M02 的 `kv_cache=None` 连续路径要保留，作为 M04/M10 的 oracle；M06 会按新签名调用。

---

## 5. 实现任务

> 按顺序做；每个任务先写失败测试，再写实现。不要一次改多个文件。

### T1 · AttentionMetadata + backend ABC（30 分钟）

文件：`nanovllm/attention/backend.py`

做：按 §4 定义 dataclass 和 ABC，`forward` 只声明；加模块内 `_validate_metadata`
（dtype、shape、`query_start_loc[0]==0`、`slot_mapping.numel()==num_actual_tokens`）；
失败测试要求 `q` fp32、`slot_mapping` int32、`block_table` int64 都触发 `AssertionError`。

验收：`python -m pytest tests/test_m04_backend.py -q`

坑：`num_actual_tokens` 不是 padding 后大小；`block_table` 宽度是 `max_blocks`；`causal`
是 bool，不是张量。

### T2 · naive backend + KV 写入（60 分钟）

文件：`nanovllm/attention/ops/cache.py`、`nanovllm/attention/backends/naive.py`

做：实现 §3.7 的 `reshape_and_cache()` 和 `NaivePagedAttention.forward`；先和“手工 gather
的 dense K/V + SDPA”对拍；用 2 请求、不同 `seq_len`、block table 带 padding 的 case 跑通。

验收：`python -m pytest tests/test_m04_naive.py -q`

坑：`k_cache[blocks]` 是 `[nblk, B, H, D]`，reshape 后必须 `[:S]`；`repeat_interleave`
不是 `repeat`；`out[qs:qe]` 赋值前要 transpose 回 `[qlen, Hq, D]`。

### T3 · 接进 Qwen3 attention 层（60 分钟）

文件：`nanovllm/models/qwen3.py`

做：按 §4 改 `Qwen3Attention` / `Qwen3Model`；每层拿自己的 `(k_cache, v_cache)`；保留
`kv_cache=None` 的 M02 路径用于对拍。

验收：`python -m pytest tests/test_m04_model_parity.py -q -k "single_request"`

坑：`positions` 只给 RoPE，不要当 attention mask；q/k/v 进 backend 前保持 `[T,H,D]`；
层号传错会读到上一层的 cache。

### T4 · 三路对拍：连续 / naive / Triton（40 分钟）

文件：`tests/test_m04_parity.py`

做：固定 seed、同一组权重；prefill 后比 logits，再 decode 5 步每步比 logits；
`atol=1e-3, rtol=1e-3`；greedy token 完全一致；M02 是最高优先级 oracle。

验收：`python -m pytest tests/test_m04_parity.py -q`

坑：不要比较 cache 内容；无 GPU 时 skip Triton；bf16 先比 logits 再比 token；三路必须用
同一份 `positions`。

### T5 · Triton decode kernel（90 分钟）

文件：`nanovllm/attention/ops/paged_attention.py`、`nanovllm/attention/backends/triton.py`

做：先只支持 decode（每请求 1 token，`query_start_loc=[0..B]`）；grid `(T,16)`；
`BLOCK_N=32`、`HEAD_DIM=128`、`num_warps=4` 起步；fp32 累加，最后 cast 回 bf16；
K/V 和 block table 的 load 都带 mask。

验收：`python -m pytest tests/test_m04_triton.py -q -k "decode"`

坑：`seq_lens` 读成 query 长度会少读 KV；`offs // BLOCK_SIZE` 用逻辑块号查 block table；
最后一个 block 的 `offs >= S` 没 mask，会读到别的请求。

### T6 · Triton prefill / chunked（60 分钟）

文件：同 T5

做：复用 T5 的 kernel，不加新分支；覆盖 `q_len=20`（整段 prefill）和
`S=16, q_len=3, q_start_in_seq=13`（chunked）；只和 naive 比数值，不看性能。

验收：`python -m pytest tests/test_m04_triton.py -q -k "prefill or chunked"`

坑：绝对位置必须用 `S - qlen + (tok - qs)`，不能把 `tok` 当序列位置；mask 必须是
`key_pos <= query_pos`，不是 `<`；整段 prefill 的 `S == qlen` 分支不能误加 mask。

### T7 · GQA + causal 单测（30 分钟）

文件：`tests/test_m04_gqa.py`

做：让 q head 0/1 完全相同，断言输出相同；让 kv0/kv1 明显不同，断言 head 0/1 跟 kv0、
head 2/3 跟 kv1；再断言 16 个 head 不全是同一个张量。

验收：`python -m pytest tests/test_m04_gqa.py -q`

坑：`kv_head = q_head // group_size` 写成 `%`；`repeat` 与 `repeat_interleave` 混用；
只比 shape 不比数值。

### T8 · benchmark（30 分钟）

文件：`benchmarks/bench_m04_attention.py`

做：decode batch=1/8/32，固定 `seq_len=1024`、`block_size=16`；warmup 10 步、计时 50 步，
`torch.cuda.synchronize()` 后读时间；输出 naive vs Triton tokens/s 和加速比，并打印
`assert_close` 结果。

验收：`python benchmarks/bench_m04_attention.py --batch-size 1 8 32`

坑：把首次编译算进计时；两后端 `seq_len` 不同；只报速度不做正确性断言。

---

## 6. 测试与对拍

### 6.1 必测 case

| case | 配置 | 断言 |
|---|---|---|
| 整段 prefill | `q_len=S=20`, `block_size=16` | naive vs Triton vs M02，`atol=rtol=1e-3` |
| decode | `q_len=1`, `S=20` | 同上 |
| chunked | `q_len=3`, `S=16`, 序列内起点 13 | 显式 mask 生效；naive vs Triton |
| 边界 | `S=1`、`S=16`、`S=17` | 最后一个 block 的 mask 正确 |
| block table padding | 宽度 8，实际只需 1–2 块 | 不读 padding 列 |
| GQA | 16 Q / 8 KV | head 0/1 同输出；head 0/2 不同 |
| 多请求 | 两个请求不同 `S`、不同 `q_len` | 扁平 batch 拼接正确 |

### 6.2 对拍顺序

顺序是：**手工 dense → naive → M02 连续 cache → Triton**。先让 naive 和 M02 一致，再跑
Triton；每一步固定 seed，失败时打印 `req`、`q_start`、`S`、`blocks`、`slot_mapping` 前 8 个值。

### 6.3 常见失败现象

| 现象 | 最可能的原因 |
|---|---|
| 第一个 token 对，后面逐渐乱 | chunked 的绝对位置用了 local index |
| 只有最后一个 block 乱 | `tl.load` 的 `offs < S` mask 没加 |
| 某些 head 全一样 / 全错 | GQA 的 `repeat` 用错，或 `kv_head` 映射反了 |
| 全 NaN | online softmax 的 `m_i` 初始化成 0，或 `l_i` 除了 0 |
| 输出 shape 对但值像别的请求 | block table padding 列被读，或 slot mapping 用了逻辑块号 |
| naive 对、Triton 错 | 先打印 Triton 的 `slot` / `blk`，再和 naive 的 `blocks` 对比 |

---

## 7. 常见坑

1. **`seq_lens` 是总长，`query_start_loc` 是前缀和。** decode 混用会少读 KV；把 `loc[r+1]-loc[r]` 当 `seq_len` 会算错 mask。
2. **chunked prefill 的 causal mask 用 `local_q_idx`。** 必须用 `S - qlen + local_q_idx`；
   或只在 `q_start_in_seq==0` 时用 `is_causal=True`。
3. **`block_table` 的 padding 行指向 0 号块。** 0 是真实块；kernel 只循环
   `ceil(seq_len/block_size)` 次，不能把整行读进来。
4. **GQA 映射写反。** `kv_head = q_head // group_size`；扩 KV 用 `repeat_interleave`。
5. **`tl.dot` 的 shape 不是 2 的幂。** nano 单 token 版本没有 `tl.dot`；改 tile 版本时
   `BLOCK_M` / `BLOCK_N` / `HEAD_DIM` 都要是 2 的幂。
6. **最后一个 block 越界。** `tl.load` 的 K/V、block table 都要 `mask=offs < S`。
7. **输出 dtype 是 fp32。** `acc` 用 fp32 没问题，写回前必须 `.to(q.dtype)`。
8. **cache 写入和读取 layout 不一致。** 统一 `[N,B,H,D]`；写用 `view(-1,H,D)`，读用
   `k_cache[blocks].reshape(-1,H,D)`。
9. **`num_actual_tokens` 与 `q.shape[0]` 混用。** M10 的 graph 会把 batch padding 到 bucket，
   T 可能大于用户请求数；kernel 的 grid 用 metadata 的 T，输出由上层截断。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m04_backend.py tests/test_m04_naive.py tests/test_m04_triton.py -q` 全绿；
- [ ] 三路（M02 / naive / Triton）在 prefill、decode、chunked 三种 case 上 logits 一致；
- [ ] GQA 单测能证明 head 0/1 共享 kv0，而不是“看起来差不多”；
- [ ] `block_table` 有 padding、`seq_len` 在 1/16/17 边界时不越界；
- [ ] `benchmarks/bench_m04_attention.py` 能输出 batch=1/8/32 的 naive vs Triton 表；
- [ ] 能解释为什么 naive 必须保留到 M10/M12；
- [ ] `PROGRESS.md` 已更新，commit：`M04: paged attention (naive + triton)`。

---

## 9. 交接：给 M05 / M06 / M10

**新增/修改文件**

```text
nanovllm/attention/{backend.py, ops/cache.py, ops/paged_attention.py,
                     backends/naive.py, backends/triton.py}
nanovllm/models/qwen3.py               # 消费 kv_caches + attn_metadata
tests/test_m04_*.py；benchmarks/bench_m04_attention.py
```

**公开 API**：§6.5 字段与签名不变；模型内部签名见 §4；没有新增公共 selector，M06 直接实例化 `TritonPagedAttention`，测试可实例化 `NaivePagedAttention`。

**已知限制**：只支持 BF16 / causal / GQA / `block_size=16` / 单 KV group；Triton 是
“一 token 一 head”的 2D grid，prefill 性能一般但正确；不做 2D/3D、segment softmax、
quant、alibi、sliding window、MLA；naive 只作 oracle；M10 的 CUDA graph 只接 Triton。

**下一个模块的入口**：M05 不需要 attention；M06 按 §3.1 填五个张量；M10 按 §3.1/§3.2
核对静态 buffer 的 shape 与 layout。

---

## 10. 自测题

1. `query_start_loc` 和 `seq_lens` 各是什么？为什么两个都需要？
2. 为什么 decode 的 causal mask 可以省掉？chunked prefill 为什么不能省？
3. chunked prefill 时，怎么算 query 的绝对位置？举一个 `S=20, q_len=3` 的例子。
4. GQA 的 16 个 Q head 怎么映射到 8 个 KV head？`repeat` 和 `repeat_interleave` 差在哪？
5. online softmax 的 `m_i` / `l_i` / `acc` 分别怎么更新？为什么 `m_i` 不能初始化为 0？

---

## 11. 教练流程（给 agent 用）

20 min 讲 §3.1 五个张量 → 30 min 带读 §2 的 1–4 / 14 条 → 60 min 写 T1/T2 → 30 min
三路对拍（先 naive vs M02）→ 90 min 写 T5/T6 → 60 min T7 GQA + 边界 → 30 min
T8 benchmark + 复盘。

禁止：先讲 Triton；直接贴 naive 完整实现；通读 `backend.py` / runner；跳过失败 case 调性能；
用 `time.time()` 不同步。**先让 naive 跑通**；用户一上来写 kernel，就让他先用 naive 打印
`blocks` / `slot_mapping` / mask，确认数据流。
