# M02 · 朴素 KV Cache 与自回归生成

> **模块目标**：在 M01 模型上加入**连续（非分页）KV cache**，实现 prefill + decode 循环，
> 第一次用 Qwen3-0.6B 生成连贯文本，并和 HF greedy 对拍。
>
> **预计工时**：3–4 小时　**依赖**：M01　**测试**：GPU
>
> **产出**：`python -m nanovllm.entrypoints.generate --prompt "The capital of France is"`
> 输出连贯文本；`pytest tests/test_m02_naive_kv.py -q -m gpu` 全绿。
>
> 这个 cache 是**临时脚手架**：M03/M04 会换成 paged，但它要保留到 M04 作为对拍 oracle。

---
## 0. 本模块结束时你会有什么

```bash
python -m nanovllm.entrypoints.generate --model models/Qwen3-0.6B \
  --prompt "The capital of France is" --max-tokens 20 --greedy
# The capital of France is Paris. ...
# prefill: 1234.5 tok/s (P=5, 4.1 ms)   decode: 45.6 tok/s (N=19, 416.7 ms)
```

以及 5 个能脱口而出的答案：prefill/decode 的 attention shape、decode 为什么不需要 causal mask、positions 怎么算、连续 cache 为什么浪费显存、chunked prefill 时 `start_pos` 怎么变。

---
## 1. 为什么需要这个模块（10 分钟）

M01 每生成一个 token 都要把整段 prompt 重算一遍：第 t 步算了 t 个 token 的 K/V，
生成 100 个 token 就是 O(T²) 次投影。KV cache 把每层已经算过的 K/V 存下来，decode
每步只算 1 个 token 的 Q/K/V，再和历史上的 K/V 做 attention。

vLLM 的 KV cache 是分页的（`KVCacheManager` + `BlockPool` + `block_table`），M03 才做。
本模块先用**每个请求一整块连续 buffer**，把“写 cache / 读 cache / 位置 / mask / 采样”
这条链路跑通。它的价值有两个：

1. 它是 M03/M04 的**数值 oracle**：paged attention 必须和它 `allclose`；
2. 它把 `num_computed_tokens` 这个 V1 核心变量具体化：M02 里它就是 `cache.length`。

本模块仍然**不做** batching、chunked prefill、prefix caching、top-k/top-p、增量 detokenize。

---
## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`。行号于 2026-10-29 校验。
> `gpu_model_runner.py` 有 7000+ 行，**不要通读**；本模块只看两个切片，合计约 150 行。

### 2.1 必读（按顺序）

| # | 位置 · 符号 | 看什么 | 可跳过 |
|---|---|---|---|
| 1 | `vllm/v1/worker/gpu_model_runner.py:1920 · _prepare_inputs()` | 只读 1920–2035：`req_indices = np.repeat(...)`、`cu_num_tokens`、`positions = num_computed_tokens_cpu[req_indices] + query_pos`、`token_indices`、`query_start_loc` | block table commit、mrope、prompt embeds、spec decode、后续 500 行 |
| 2 | `vllm/v1/worker/gpu_model_runner.py:2191 · logits_indices` | 读 2180–2200：`logits_indices = query_start_loc[1:] - 1`；注释说 chunked prefill 的 partial request 也会采样、之后被忽略 | spec decode 分支 |
| 3 | `vllm/v1/worker/gpu_model_runner.py:3940 · execute_model()` | 只读 3940–4035：`_update_states` → `_prepare_inputs` → forward 的骨架；理解 `execute_model` 只前向、`sample_tokens` 另调 | CUDA graph、DP、EC transfer、kv connector、后续几千行 |
| 4 | `vllm/v1/worker/gpu_input_batch.py:35 · CachedRequestState` | 字段 `prompt_token_ids`、`output_token_ids`、`num_computed_tokens`、`block_ids`；`num_tokens` property | mm / LoRA / pooling / mrope 字段 |
| 5 | `vllm/v1/worker/gpu_input_batch.py:345 · InputBatch.add_request()` | 只读 345–400：`token_ids_cpu` 怎么写 prompt+output、`num_computed_tokens_cpu`、`block_table.add_row` | 采样参数注册、spec token、replay 分支 |
| 6 | `vllm/v1/request.py:272 · append_output_token_ids()` | 同时改 `_output_token_ids` 和 `_all_token_ids`，然后 `update_block_hashes()` | `update_block_hashes` 内部（M07） |
| 7 | `vllm/model_executor/layers/attention/attention.py:229 · Attention` / `vllm/model_executor/layers/attention/attention.py:488 · forward()` | k/v 如何按 `slot_mapping` 写进 cache、再读出来做 attention；**只读 forward 开头 40 行** | backend dispatch、quant、CUDA graph、MLA |
| 8 | 选读 `vllm/v1/worker/gpu_model_runner.py:1172 · _update_states()` | 只读开头 30 行：finished → `requests.pop` / `input_batch.remove_request`；理解“请求状态在 worker 侧也有一份” | 后面所有分支 |

### 2.2 建议读（选读）

- HF `DynamicCache` / `past_key_values` 的**行为**（不是实现）：每层存 K/V，decode 时
  追加；可以用 `use_cache=True` 的 `generate` 作为 oracle。
- `vllm/v1/request.py:299 · num_tokens_with_spec`：M02 不需要，但要知道 V1 的调度目标是它。
- `nanovllm/sampling_params.py` 的 `is_greedy` / `eos_token_id` / `max_tokens`：M02 只读这三个。

### 2.3 明确跳过

CUDA graph、spec decode、mrope、KV connector、DP/PP、`initialize_kv_cache`
（`vllm/v1/worker/gpu_model_runner.py:7055`，paged 版，M03 再看）、`block_table` / `slot_mapping`
（M03）、`Sampler`（M08）、`IncrementalDetokenizer`（M08）。**第一遍全部不看。**

读完 §2 你应该能回答：vLLM 怎么把“多个请求的 token”拼成一个 batch？`positions` 为什么
是 `num_computed_tokens + query_pos`？`logits_indices` 为什么是 `query_start_loc[1:]-1`？

---
## 3. 核心概念

### 3.1 prefill vs decode

| 阶段 | `input_ids` | `positions` | q | k/v（本步） | cache 前 | cache 后 | 取哪行 logits |
|---|---|---|---|---|---|---|---|
| prefill | `[P]` | `[0..P-1]` | `[P,16,128]` | `[P,8,128]` | 0 | P | `[-1]` |
| decode 第 t 步 | `[1]` | `[n]`（n=cache 长度） | `[1,16,128]` | `[1,8,128]` | n | n+1 | `[0]` 或 `[-1]` |

prefill 一次算完 prompt，建立 KV，采样出第一个 output token；decode 每步只喂上一步
采样出的 1 个 token，位置是**绝对位置** `n`，不是 0。

### 3.2 cache 形状：本模块定死一种

```text
每层一对：k_cache, v_cache : [max_len, num_kv_heads, head_dim]   # [4096, 8, 128]
ContiguousKVCache 内部：list[tuple[k_cache, v_cache]]，28 层
get(layer_idx, length) -> [1, num_kv_heads, length, head_dim]
```

为什么用 `[max_len, H, D]` 而不是 `[1, H, max_len, D]`？因为 append 是**按位置写**
（`k_cache[start:start+T] = k`），第一维是时间，写起来最自然；`get` 时再 `transpose(0,1).unsqueeze(0).contiguous()` 成 `[1,H,L,D]`；这次 copy 是
连续 cache 的固有开销，也是 paged attention 想省掉的东西之一。

**容量**：每 token 每层 `2 × 8 × 128 × 2 B = 4096 B`，28 层共 `112 KiB/token`。
`max_len=4096` 时每请求 `448 MiB`；8 个并发请求就是 `3.5 GiB`。而如果每个请求实际只用
512 token，真实需要只有 `448 MiB`，**浪费 8 倍**。M03 用 block（16 token）分配把这
个浪费降到每请求最多 15 个 token 的 slack。

### 3.3 positions 是绝对位置

```text
prefill: positions = arange(0, P)                  # start_pos = 0
decode : positions = tensor([cache.length])        # start_pos = cache.length
```

`cache.length` 就是 V1 的 `num_computed_tokens`：**已经算进 KV 的 token 数**。
prefill 后 `length=P`，此时还没把采样出的 `y0` 算进去；decode 第一步把 `y0` 放在位置 P，
追加后 `length=P+1`，采样出 `y1`。所以第 k 个 output（k 从 0 计）对应位置 `P+k-1`。

在 attention 里断言 `positions == arange(start, start+T)`，并要求 `T == 1 or start == 0`
（M02 不支持 chunked prefill）。M05 做 chunked 时这个断言会变成显式 mask 的起点。

### 3.4 attention mask

```text
prefill: q_len = T, kv_len = T, start = 0   -> SDPA(is_causal=True)
decode : q_len = 1, kv_len = L, start = L-1 -> SDPA(is_causal=False)，不需要 mask
```

**decode 不需要 causal mask**：唯一的 query 可以看全部历史，没有“未来”可遮。
这里有一个很容易踩的 PyTorch 语义：`F.scaled_dot_product_attention(..., is_causal=True)`
在 `q_len != kv_len` 时按**左上角对齐**，`q_len=1, kv_len=L` 会只允许 key 0，
输出完全错。所以 decode 必须 `is_causal=False`（或显式构造右下角对齐的 mask）。

### 3.5 只取最后一个 token 的 logits

nano 的 logits 是 `[T, vocab]`，没有 batch 维：

- prefill：`logits[-1]` 是最后一个 prompt token 的 logits → 采样 `y0`；
- decode：`logits[0]`（也是 `[-1]`）→ 采样下一个 token。

不要写 `logits[:, -1]`：那是 `[T, vocab]` 的最后一**列**，shape 变成 `[T]`，采样器会
拿到 T 个“词表索引”。也不要把整个 `[T, vocab]` 喂给采样器。

### 3.6 采样最小版

```python
def sample_token(logits_last, params, generator):
    if params.is_greedy: return int(logits_last.float().argmax())
    probs = torch.softmax(logits_last.float() / params.temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
```

M02 只实现 greedy + temperature；`top_k/top_p/min_p/penalty/logprobs` 留到 M08。
`eos_token_id`、`ignore_eos`、`stop_token_ids`、`max_tokens` 用于停止。
greedy 在 float32 上 argmax，避免 bf16 的并列值导致和 HF 差一个 token。

### 3.7 detokenize 最小版

```python
text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
```

**每步都 decode 完整的 `output_token_ids`**，不要逐个 token `decode([t])`：BPE 的一个
字符可能由多个 token 组成，单独 decode 会得到替换字符。增量 detokenize 是 M08 的事。

---
## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `_prepare_inputs` 的 `req_indices` / `cu_num_tokens` | 不需要（batch=1） | 删；只保留 `start_pos` / `positions` |
| `positions = num_computed_tokens + query_pos` | `positions = arange(start, start+T)` | 简化；`start = cache.length` |
| `logits_indices = query_start_loc[1:]-1` | `logits[-1]` | 单请求取最后一行 |
| `CachedRequestState` | 局部变量 `prompt_ids` / `output_ids` / `cache` | 删 `block_ids` / mm / LoRA |
| `InputBatch.token_ids_cpu` | Python list `prompt + output` | 不预分配 |
| `Request.append_output_token_ids()` | `output_ids.append(t)`；`cache.length` 在下次 forward 时 +1 | 无 block hash（M07） |
| `Attention.forward` 按 `slot_mapping` 写 cache | `ContiguousKVCache.append` 按下标写 | 连续 buffer，无 slot mapping |
| HF `DynamicCache` | 对拍 oracle | 不实现 HF 接口 |

**本模块新增的接口（非冻结，不进 `docs/00_architecture.md` §6）**：

```python
# nanovllm/models/kv_cache.py
class ContiguousKVCache:
    def __init__(self, num_layers, max_len, num_kv_heads, head_dim,
                 dtype=torch.bfloat16, device="cuda"): ...
    def append(self, layer_idx, k, v, start_pos): ...   # k/v: [T, Hkv, D]
    def get(self, layer_idx, length): ...               # -> [1, Hkv, length, D]
    def reset(self): ...                                # length = 0
    @property
    def length(self) -> int: ...                        # = num_computed_tokens

# nanovllm/models/qwen3.py（向后兼容：默认 None 走 M01 路径）
Qwen3Attention.forward(hidden_states, positions, kv_cache=None, layer_idx=0)
Qwen3Model.forward(input_ids, positions, kv_cache=None)
Qwen3ForCausalLM.forward(input_ids, positions, kv_cache=None)

# nanovllm/entrypoints/generate.py
@dataclass
class GenerationResult:
    output_token_ids: list[int]; text: str; finish_reason: str
    prefill_time_s: float; decode_time_s: float
def generate(model, tokenizer, prompt: str, params: SamplingParams,
             max_model_len: int = 4096) -> GenerationResult: ...
```

约定：① `kv_cache` 传 `ContiguousKVCache` 实例，不传 list；它内部就是每层一对
`[max_len, Hkv, D]`，`append` 需要 `layer_idx` 和统一的 `length`；
② `append` 用 `self.length = max(self.length, start_pos + T)`，被 28 层各调一次也不会
重复累加；③ `length` 只增不减，`reset()` 归零但不擦显存（读只看 `[:length]`，安全）。

---
## 5. 实现任务

> 顺序：cache → attention 接 cache → prefill → decode 循环 → 采样 → tokenizer/CLI → 对拍。
> 先 batch=1、greedy，跑通再谈 temperature；不要一开始就做 batching。

### T1 · `ContiguousKVCache`（25 分钟）

文件：`nanovllm/models/kv_cache.py`。按 §3.2 分配 28 对 `[max_len, 8, 128]`；
`append` 写入并 `length=max(...)`；`get` 返回 `[1,8,length,128]`（`.contiguous()`）；
`reset` 只把 `length=0`。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -k cache
```

失败测试先写：

```python
c = ContiguousKVCache(2, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(5, 8, 128, dtype=torch.bfloat16, device="cuda")
for layer in (0, 1): c.append(layer, k, k, 0)
assert c.length == 5 and c.get(0, 5).shape == (1, 8, 5, 128)
```

坑：`append` 每层都 `length += T`（28 倍）；`get` 忘了 transpose，返回 `[1,L,8,128]`；
`max_len` 越界不报错，写到下一层。

### T2 · attention 接 cache（40 分钟）

文件：`nanovllm/models/qwen3.py`。`Qwen3Attention.forward` 增加 `kv_cache=None,
layer_idx=0`：`start_pos = int(positions[0])`；若 `kv_cache is not None`，先
`append(layer_idx, k, v, start_pos)`，再 `get(layer_idx, start_pos+T)`，再 repeat。
`Qwen3Model.forward` 把 `kv_cache` 和 `layer_idx` 逐层传下去。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -k "attention and cache"
```

测试：同一输入、cache 为空时，带 cache 的 prefill logits 与 M01 无 cache 路径
`assert_close`（`atol=2e-2`）；再断言 cache 写完后 `length == P`。
坑：把 cache 拼到 q 的序列维（q 的 T 不变）；decode 时 `start_pos` 用了 0；
`repeat_interleave` 前没把 cache 转成 `[1,H,L,D]`。

### T3 · prefill（25 分钟）

文件：`nanovllm/entrypoints/generate.py`。`generate` 第一步：`positions =
arange(P)`，`cache = ContiguousKVCache(...)`，`logits = model(ids, positions,
cache)`，取 `logits[-1]` 采样 `y0`，追加到 output。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -k prefill
```

断言：prefill 后 `cache.length == P`；`y0` 与 M01 无 cache 的 greedy 首 token 相同。
坑：prefill 后把 `length` 又加了 1（`y0` 还没算进 KV）；`positions` 用了 `arange(1,P+1)`；
忘了 `torch.no_grad()`。

### T4 · decode 循环（40 分钟）

文件：`nanovllm/entrypoints/generate.py`。每步：`pos = cache.length`；`logits =
model(tensor([last]), tensor([pos]), cache)`；`next = sample(logits[-1])`；
`output.append(next)`；先 append 再判 eos / `max_tokens` / `stop_token_ids`（eos 保留，和 HF 一致）。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -k decode
```

测试：生成 20 个 token 不崩；每步后 `cache.length == P + step`（step 从 1 计）；把某步的 `length`
打印出来人工核对。坑：position 用了循环变量 `step` 而不是 `cache.length`；
decode 输入用了 `output[-2]`（应喂 `output[-1]`）；忘记在采样后判断 eos。

### T5 · greedy / temperature（20 分钟）

文件：`nanovllm/entrypoints/generate.py`（或 `nanovllm/sample/sampler.py` 先放一个
`sample_token`）。按 §3.6；`temperature=0` 走 argmax，`>0` 走 softmax+multinomial；
`torch.Generator(device="cuda").manual_seed(params.seed)`。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -k sampling
```

测试：同一 seed 两次采样序列相同；greedy 两次相同；`temperature=1.0` 能跑通（统计正确性
留给 M08）。坑：对 bf16 logits 直接 argmax；`multinomial` 没传 generator；softmax 前
忘了除 temperature。

### T6 · tokenizer / detokenizer（20 分钟）

文件：`nanovllm/entrypoints/generate.py`。`AutoTokenizer.from_pretrained`；
`prompt_ids = tokenizer(prompt).input_ids`；生成结束 `tokenizer.decode(output_ids,
skip_special_tokens=True)`。验收：

```bash
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('models/Qwen3-0.6B'); print(t.decode(t('hello').input_ids))"
```

以及 T7 的 CLI 能打印可读文本。坑：逐个 token decode（乱码）；忘了 `skip_special_tokens`
把 `<|im_end|>` 打出来；tokenizer 默认不加 BOS，不要手动加。

### T7 · CLI（25 分钟）

文件：`nanovllm/entrypoints/generate.py` + `nanovllm/entrypoints/__init__.py`，再建一个
三行的 `nanovllm/generate.py` shim（`from .entrypoints.generate import main; main()`），
让 ROADMAP 里的 `python -m nanovllm.generate` 也能跑。
`argparse`：`--model`、`--prompt`、`--max-tokens`、`--temperature`、`--seed`、
`--max-model-len`。打印生成文本 + 分段 tokens/s。验收：

```bash
python -m nanovllm.entrypoints.generate --model models/Qwen3-0.6B \
  --prompt "The capital of France is" --max-tokens 20 --greedy
```

应出现 `Paris`；若乱码，先回 T4 检查 positions 和 mask。坑：`max_tokens` 没和
`max_model_len - P` 取 min；CLI 没设 `eos_token_id`（`SamplingParams` 默认 None，
必须从 tokenizer 填）；规定 `--greedy` 优先于 `--temperature`。

### T8 · HF 对拍 + tokens/s（30 分钟）

文件：`tests/test_m02_naive_kv.py`、`scripts/check_generation_parity.py`（可选）。

```python
hf = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
hf_ids = hf.generate(ids, do_sample=False, max_new_tokens=20,
                     pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
assert result.output_token_ids == hf_ids
```

`tokens/s`：prefill 用 `P / prefill_time`；decode 用 `(N-1) / decode_time`
（首 token 是 prefill 产出的）。计时前后 `torch.cuda.synchronize()`。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -k "hf or parity"
```

坑：HF 默认 sdpa；忘了 `do_sample=False`；只比文本不比 token（空格/特殊 token 会掩盖
差异）；计时没 synchronize，数字大 10 倍。

### T9 · 测试文件（25 分钟）

文件：`tests/test_m02_naive_kv.py`。覆盖：cache shape、prefill 与 M01 一致、20-token
HF 对拍、prompt 长度 `1/15/16/17/100`、EOS/max_tokens 停止、seed 复现、tokens/s 打印。
GPU 测试标 `@pytest.mark.gpu`，模型不存在 `skip`。验收：

```bash
python -m pytest tests/test_m02_naive_kv.py -q -m gpu
python -m pytest tests -q          # M00/M01 的 CPU 测试不能回归
```

坑：5 个长度都跑 HF 太慢（只对 1/17 做对拍，其余 smoke）；每次测试新建 cache 避免
`length` 串味；`max_len` 用 128 就够测 shape，别分配 4096。

---
## 6. 测试与对拍

**prefill 一致性**：同一模型、同一 `ids`，一次 `kv_cache=None`，一次新建 cache；
比 `logits[-1]`，`atol=2e-2, rtol=1e-2`。这是“cache 读写没改变数学”的基线。

**HF greedy 20 token**：

```python
ids = torch.tensor([tok("The capital of France is").input_ids], device="cuda")
out = hf.generate(ids, do_sample=False, max_new_tokens=20, pad_token_id=tok.eos_token_id)
assert result.output_token_ids == out[0, ids.shape[1]:].tolist()
```

**逐 token logits 对拍**：如果 token 序列不一致，先别怀疑采样。把 HF 的完整序列每步
重算一遍（不用 `past_key_values` API，避免版本差异），取 `logits[0, -1]` 和 nano 的
`logits[-1]` 比；第一处差 > 2e-2 的 step 就是 bug 所在。

**长度边界**：`P = 1, 15, 16, 17, 100`，`max_tokens=8`；断言不崩、长度正确（16 是 M03 的 block_size）。

**停止条件**：单元测试直接喂 fake logits / fake token 流验证 eos、`stop_token_ids`、
`max_tokens`、`ignore_eos`；端到端只断言 `finish_reason ∈ {stop, length}`，不要“祈祷”
模型自然吐 EOS。

**性能**：打印 `prefill tok/s` 和 `decode tok/s`（分开），并记录 P/N。M02 不追求数字，
但要建立“decode 比 prefill 慢很多”的直觉（每步只 1 token，kernel launch 占比高）。

**失败现象 → 原因**：

| 现象 | 原因 |
|---|---|
| 第二个 token 开始重复/乱码 | cache 没 append、`start_pos` 错、position 用了 0 |
| prefill 正常，decode shape 报错 | decode 的 q 用了 `[T,...]`，或 cache `[L,H,D]` 没转 `[1,H,L,D]` |
| decode 输出只看第一个 token | `is_causal=True` 且 `q_len=1, kv_len=L` |
| 和 HF 从第一个 token 就不同 | 采样/positions/prefill mask；先比 logits |
| tokens/s 高得离谱 | 没 `torch.cuda.synchronize()` |

---
## 7. 常见坑

1. **position 用 chunk 内偏移**：decode 的 position 必须是 `cache.length`
   （`P, P+1, ...`），不是 0/1/2。错了短 prompt 还能看，长 prompt 直接崩。
2. **`start_pos` 算错**：prefill 用 0，decode 用 `cache.length`（= 已算进 KV 的 token 数）。
   prefill 后不要 +1，采样出的 token 还没进 cache。
3. **decode 把 cache 拼进 q 的序列维**：q 始终是 `[1, 16, 128]`（T=1）；cache 只影响
   k/v 的 `kv_len`。拼错会得到 `[1,16,L,128]` 的 q，shape 对不上或结果全错。
4. **`is_causal=True` 配 `q_len=1`**：SDPA 按左上角对齐，只允许 key 0。decode 必须
   `is_causal=False`；prefill 才能 `True`。见 §3.4。
5. **把 `[T, vocab]` 整个喂给采样器**：应该 `logits[-1]`；写成 `logits[:, -1]` 会得到
   `[T]`，采样出的是“位置索引”而不是 token。
6. **逐个 token decode**：BPE 多 token 字符会变成替换符；decode 整个 output list。
7. **`append` 每层都加 `length`**：28 层调用 28 次，`length` 变 28 倍；用 `max` 或只在
   layer 0 加。M02 用 `max`。
8. **bf16 greedy 差一个 token**：先比 logits；argmax 在 float32 上做；HF 也可能因为
   后端不同差 1 ulp，不要只比 token。计时前后记得 `torch.cuda.synchronize()`，否则
   `perf_counter` 量到的是 launch 时间。

---
## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m02_naive_kv.py -q -m gpu` 全绿；
- [ ] `pytest tests -q` 不回归（M00/M01 的 CPU 测试仍绿）；
- [ ] prefill logits 与 M01 无 cache 版本一致（`atol=2e-2`）；
- [ ] 20-token greedy 与 HF `generate(do_sample=False)` **token 完全一致**；
- [ ] prompt 长度 `1/15/16/17/100` 都不崩；
- [ ] CLI 能打印 `Paris` 与分段 tokens/s；
- [ ] 能算出 4096 上下文、8 并发下连续 cache 的显存占用（≈3.5 GiB）并解释浪费；
- [ ] `PROGRESS.md` 已更新；commit：`M02: contiguous KV cache + naive generation`。

---
## 9. 交接：给 M03 的 agent

**新增文件**：`nanovllm/models/kv_cache.py`、`nanovllm/entrypoints/generate.py`、`nanovllm/entrypoints/__init__.py`、`tests/test_m02_naive_kv.py`。

**公开 API（临时）**：

```python
ContiguousKVCache(num_layers, max_len, num_kv_heads, head_dim, dtype, device)
cache.append(layer_idx, k, v, start_pos); cache.get(layer_idx, length)
cache.reset(); cache.length
Qwen3ForCausalLM.forward(input_ids, positions, kv_cache=None)
generate(model, tokenizer, prompt, params, max_model_len=4096) -> GenerationResult
```

**必须保留**：`ContiguousKVCache` 不删。M04 的 naive/Triton paged 后端要以它为
oracle；M03 做块分配时，先用它验证“同样的 KV 值，连续 vs 分页结果一致”。

**已知限制**：batch=1；无 chunked prefill（`T>1` 时 `start_pos` 必须为 0）；无 prefix
caching；无 top-k/top-p/penalty/logprobs（M08）；无 stop string；连续 buffer 浪费显存。

**M03 入口**：把 `[max_len, Hkv, D]` 的连续 buffer 换成 `[num_blocks, block_size,
Hkv, D]` 的 block pool；`cache.length` 变成 `Request.num_computed_tokens`；
`append(start_pos)` 变成 `slot_mapping` 写入。接口形状参考 `docs/00_architecture.md` §6.5。

---
## 10. 自测题

1. prefill 和 decode 的 attention 输入 shape 分别是什么？`q_len` 和 `kv_len` 各是多少？
2. 为什么 decode 不需要 causal mask？`is_causal=True` 在 `q_len=1` 时会怎样？
3. `positions` 在 prefill 和 decode 时分别怎么算？prefill 后 `cache.length` 是多少，
   为什么不是 `P+1`？
4. 连续 KV cache 为什么浪费显存？用 16GB / Qwen3-0.6B / 4096 上下文 / 8 并发举一个数。
5. 如果 prefill 分块（chunked），cache 的 `start_pos` 和 mask 要怎么变？

---
## 11. 教练流程（给 agent 用）

| 时间 | 做什么 | 检查点 |
|---|---|---|
| 15 min | 讲 §3.1/§3.2 的 shape 与 `length` 语义 | 能说出 prefill 后 `length=P` |
| 25 min | 带读 `_prepare_inputs` 的 positions / `logits_indices` 两段 | 能解释 `query_start_loc[1:]-1` |
| 20 min | 用户设计 cache 形状与生命周期（纸面） | 画出 `[4096,8,128]` 与 `[1,8,L,128]` |
| 40 min | 用户写 T1/T2，先跑 shape 测试再对 logits | cache 测试绿、prefill 与 M01 一致 |
| 40 min | 用户写 T3/T4/T5，先 greedy 跑通 | 能生成 20 个 token |
| 40 min | T8 HF 对拍 + 逐 token logits debug | 20 token 完全一致 |
| 15 min | 自测题 + 更新 `PROGRESS.md` + commit | 5 题能答 |

**禁止**：一上来做 batching / chunked prefill；先 batch=1、greedy。不要直接贴完整
`generate()`；先让用户写 cache 和 attention 的 shape。不要用“模型输出看起来对”代替
HF token 对拍；不要建议通读 `gpu_model_runner.py`。
