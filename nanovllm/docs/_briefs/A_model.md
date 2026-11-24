# 文档作者 brief A：M01 模型 + M02 朴素 KV

## 通用要求（必须先读）

工作区：`/home/dictator/ai_infra/nanovllm`

先读：

1. `docs/00_architecture.md`（尤其 §5–§8：目录、冻结接口、简化清单）
2. `docs/00_facts.md`（环境、Qwen3 config、已验证行号、写作约定）
3. `docs/M00_contracts.md`（风格样例：目标 / 阅读地图 / 概念 / 任务 / 坑 / DoD / 自测 / 教练流程）
4. `docs/_template.md`
5. `ROADMAP.md`

只写这两个文件：

- `docs/M01_model.md`
- `docs/M02_naive_kv.md`

**不要修改任何代码**，不要改 `00_architecture.md` / `00_facts.md` / `M00_contracts.md`。

每个文件 320–480 行。中文，代码标识符和路径保留英文。风格要像 M00：具体、可执行、
每段都有“你应该能回答什么”。不要复制 `00_architecture.md` 的大段内容，引用它即可。

**引用规则**：vLLM 引用写到 `vllm/model_executor/models/qwen3.py:65 · class Qwen3Attention`。
下面给的行号已校验；如果你新增引用，先 `cd target/vllm-0.32.0 && grep -n "<symbol>" <file>`
确认。不确定就只写符号、不写行号。行号漂了不可怕，编造行号很可怕。

**不要**建议用户安装 vLLM 本体，不要建议通读整个 `gpu_model_runner.py`。

---

## M01：Qwen3 模型前向与权重加载

### 目标

用户用纯 PyTorch 实现 Qwen3-0.6B 的 forward + safetensors 权重加载，和 HuggingFace
`transformers` 的 `Qwen3ForCausalLM` 对拍 logits / greedy token。

### 必须覆盖的 vLLM 阅读

- `vllm/model_executor/models/qwen3.py:65 · Qwen3Attention`
  - `:156 · forward()`：QKV split、q_norm/k_norm、RoPE、`self.attn(q,k,v)`
- `vllm/model_executor/models/qwen3.py:176 · Qwen3DecoderLayer`
  - `:231 · forward()`：pre-norm residual、attention、MLP
- `vllm/model_executor/models/qwen3.py:280 · Qwen3ForCausalLM`
  - `:327 · forward()`、`:339 · compute_logits()`、`:346 · load_weights()`
- `vllm/model_executor/models/qwen2.py:81 · Qwen2MLP`
  - `:111 · forward()`：gate_up + SiLU + down
- `vllm/model_executor/models/qwen2.py:324 · Qwen2Model`
  - `:402 · forward()`：embedding → layers → final norm（只看骨架）
- `vllm/model_executor/layers/layernorm.py:37 · RMSNorm`
  - `:70 · forward_native()`：`x * rsqrt(mean(x^2)+eps) * weight`
- `vllm/model_executor/layers/rotary_embedding/__init__.py:32 · get_rope`
  - 只需知道它根据 `rope_theta` / `rope_scaling` 造 RoPE；nano 只支持默认 RoPE
- 权重名：`vllm/model_executor/models/qwen3.py:346` 的 `load_weights` 只是委托给
  `AutoWeightsLoader`；真正的映射在 `packed_modules_mapping`
  （`qkv_proj` ← q/k/v，`gate_up_proj` ← gate/up）。nano 不做 fused，按 HF 名逐个加载。

建议的阅读顺序：先看 qwen3 的 Attention/DecoderLayer 结构，再看 Qwen2MLP，最后看
RMSNorm 公式。RoPE 不要读 vLLM 的 1000 行实现；看 `RotaryEmbedding.forward_native` 或
直接看公式。

### 必须讲清的概念

1. **RMSNorm**：和 LayerNorm 的区别（不减均值、无 bias）；float32 计算再 cast 回 bf16。
2. **RoPE**：`inv_freq = 1/(theta^(2i/d))`；`rotate_half`；q/k 都加；positions 是绝对位置。
3. **GQA**：16 个 Q head，8 个 KV head；每个 KV head 被 2 个 Q head 共享。
4. **QK-Norm**：Qwen3 在 attention 前对每个 head 的 128 维做 RMSNorm；权重名
   `q_norm.weight` / `k_norm.weight`；容易漏。
5. **SwiGLU**：`down(silu(gate(x)) * up(x))`。
6. **tied embeddings**：`tie_word_embeddings=true`，`lm_head.weight` 可能不存在；用
   `embed_tokens.weight` 作为 lm_head。
7. **pre-norm residual**：每层两个 residual；HF 的实现是 `hidden = residual + attn(norm(hidden))`。

### 必须实现的接口

建议目录：

```text
nanovllm/models/qwen3.py
nanovllm/models/weight_loader.py
scripts/check_model_parity.py
tests/test_m01_model.py
```

接口（可微调，但要在文档里写清楚）：

```python
class Qwen3Config:  # 从 HF config.json 读取必要字段
    ...

class RMSNorm(nn.Module):
    def forward(self, x): ...

class RotaryEmbedding(nn.Module):
    def forward(self, positions, q, k): ...  # 原地/返回 q,k

class Qwen3MLP(nn.Module): ...
class Qwen3Attention(nn.Module):
    def forward(self, hidden_states, positions): ...
class Qwen3DecoderLayer(nn.Module): ...
class Qwen3Model(nn.Module):
    def forward(self, input_ids, positions): ...
class Qwen3ForCausalLM(nn.Module):
    def forward(self, input_ids, positions): ...
    def compute_logits(self, hidden_states): ...
    def load_weights(self, model_dir): ...
```

M01 的 attention 先**不带 KV cache**，就是标准 causal attention；M02 再加 cache。

### 实现任务（写进文档，按顺序，每个带验收）

T1 读 config + 建模型骨架；T2 RMSNorm；T3 RoPE；T4 MLP；T5 attention（无 cache）；
T6 decoder layer / model / lm_head；T7 safetensors loader（分片/index、tied embedding）；
T8 parity 脚本；T9 测试。

### 测试与对拍（必须具体）

- 固定 prompt（例如 `"The capital of France is"`），tokenize 后取 logits；
- HF oracle：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, attn_implementation="eager"
).cuda().eval()
```

- 对拍 `logits[:, -1, :]`：`torch.testing.assert_close`，bf16 建议 `atol=2e-2, rtol=1e-2`
  （先说明这是经验值，实际以你的实现为准；如果差太多，逐层比 hidden states）。
- greedy 16 token 完全一致。
- 逐层对拍：embedding、layer0 hidden、final norm、logits；定位第一处偏差。
- 注意 HF 可能默认 `sdpa`，要显式 `eager` 才有可比性。

### 常见坑（至少写这些）

- `head_dim=128` 是 config 显式值，不是 `hidden_size // num_heads = 64`；用错直接全错。
- q_norm/k_norm 是 per-head RMSNorm（对 128 维），不是对整个 hidden。
- RoPE 的 `cos/sin` dtype 要和 q 一致；不要在 float32 里算完忘了 cast。
- `rotate_half` 是前后半交换，不是相邻两两交换。
- safetensors 分片要读 `model.safetensors.index.json`；不要假设单文件。
- `lm_head.weight` 缺失时 tie；不要随机初始化。
- bf16 对拍不要用 `torch.equal`。

### 自测题

1. 为什么 Qwen3 的 `head_dim` 不能从 `hidden_size // num_heads` 推？
2. q_norm/k_norm 加在 RoPE 前还是后？为什么？
3. GQA 里 16 个 Q head 怎么映射到 8 个 KV head？
4. tied embedding 在加载权重时怎么处理？
5. pre-norm residual 的两次加法分别在哪？

### 教练流程

概念 20min → 读 qwen3.py 的 Attention/DecoderLayer 30min → 写 RMSNorm/RoPE/MLP 40min →
写 attention/model 60min → loader 30min → parity/debug 40min → 复盘 15min。
不要一上来贴完整模型代码；先让用户写 RMSNorm 和 RoPE，用 HF 单层对拍。

---

## M02：朴素 KV Cache 与自回归生成

### 目标

在 M01 模型上加入**连续（非分页）KV cache**，实现 prefill + decode 循环，第一次用
Qwen3-0.6B 生成连贯文本；和 HF greedy 对拍。这个模块的 cache 是临时脚手架，M03/M04
会替换成 paged；但它是后面所有对拍的 oracle。

### 必须覆盖的 vLLM 阅读

- `vllm/v1/worker/gpu_model_runner.py:1920 · _prepare_inputs`
  - `:1920-2020`：req_indices、cu_num_tokens、positions、token_indices、query_start_loc
  - `:2180-2200`：`logits_indices = query_start_loc[1:] - 1`
  - 只看“怎么把多个请求的 token 拼成一个 batch”；跳过 CUDA graph/spec/mrope
- `vllm/v1/worker/gpu_model_runner.py:3940 · execute_model` 的开头
  - 只看 `_update_states` → `_prepare_inputs` → forward 的骨架
- `vllm/v1/worker/gpu_input_batch.py:35 · CachedRequestState`
  - 字段：`prompt_token_ids`、`output_token_ids`、`num_computed_tokens`、`block_ids`
- `vllm/v1/worker/gpu_input_batch.py:345 · InputBatch.add_request`
  - 只看 token_ids_cpu / num_computed_tokens / block_table 的写入
- `vllm/v1/request.py:272 · append_output_token_ids`
- HF `past_key_values` / `DynamicCache` 的行为（作为 oracle）
- 可选：`vllm/attention/layer.py` 的 `Attention.forward`，看 k/v 如何写 cache

### 必须讲清的概念

1. **prefill vs decode**：prefill 一次算 prompt，decode 每步算 1 个 token。
2. **KV cache 的形状**：`[num_layers, 2, max_len, num_kv_heads, head_dim]` 或 per-layer；
   每个请求一份连续 buffer。
3. **positions 是绝对位置**：decode 第 t 步的 position 是 `prompt_len + t`，不是 0。
4. **attention mask**：prefill 是 causal；decode 是单 query 对全部历史，不需要 mask。
5. **只取最后一个 token 的 logits**：prefill 时 `logits[:, -1, :]`；decode 时唯一一个。
6. **采样最小版**：greedy + temperature；完整采样 M08 再做。
7. **detokenize 最小版**：`tokenizer.decode(output_ids, skip_special_tokens=True)`。

### 必须实现的接口

```text
nanovllm/models/qwen3.py        # 增加 use_cache / kv_cache 参数
nanovllm/worker/naive_kv_cache.py  # 或 nanovllm/models/kv_cache.py
nanovllm/engine/simple_generate.py # 或 entrypoints/generate.py
tests/test_m02_naive_kv.py
```

接口建议：

```python
class ContiguousKVCache:
    def __init__(self, num_layers, max_len, num_kv_heads, head_dim, dtype, device): ...
    def append(self, layer_idx, k, v, start_pos): ...
    def get(self, layer_idx, length): ...  # 返回 [1, heads, length, dim]
    def reset(self): ...

# Qwen3Attention.forward(hidden, positions, kv_cache=None, layer_idx=0)
```

`Qwen3Model.forward(input_ids, positions, kv_caches=None)`；`kv_caches` 是每层一个
`(k_cache, v_cache)`，形状 `[1, num_kv_heads, max_len, head_dim]`（或 `[max_len, ...]`，
文档里定死一种）。

### 实现任务

T1 设计 cache 形状/生命周期；T2 改 attention 支持 cache；T3 prefill；T4 decode 循环；
T5 greedy/temperature；T6 tokenizer/detokenizer；T7 CLI；T8 HF 对拍；T9 测试。

### 测试与对拍

- prefill logits 与 M01 无 cache 版本一致；
- 生成 20 个 token，与 HF `generate(do_sample=False)` 完全一致；
- 不同 prompt 长度（1、15、16、17、100）都不崩；
- 到 EOS 停止；`max_tokens` 停止；
- 打印 tokens/s（prefill/decode 分开）。

### 常见坑

- position 用了 chunk 内偏移，而不是绝对位置；
- cache 的 `start_pos` 算错（prefill 用 0，decode 用 `num_computed_tokens`）；
- decode 时把整个 cache 拼进 q 的 sequence 维，shape 错；
- prefill 的 causal mask 用了 `torch.triu` 但没处理 batch/head 维；
- 只取最后一个 token 的 logits，却把整个 `[T, vocab]` 都喂给采样；
- `tokenizer.decode` 每步全量 decode，出现乱码（M08 再优化增量）；
- bf16 下 greedy 对拍偶尔差一个 token：先比 logits，不要只比 token。

### 自测题

1. prefill 和 decode 的 attention 输入 shape 分别是什么？
2. 为什么 decode 不需要 causal mask？
3. `positions` 在 prefill 和 decode 时分别怎么算？
4. 连续 KV cache 为什么浪费显存？用 16GB / Qwen3-0.6B 举一个数。
5. 如果 prefill 分块（chunked），cache 的 `start_pos` 怎么变？

### 教练流程

概念 15min → 读 `_prepare_inputs` 的 positions/logits_indices 25min → 设计 cache 20min →
写 attention cache 40min → 写 generate 40min → 对拍/debug 40min → 复盘 15min。
先让用户用 batch=1、greedy 跑通，再谈 temperature；不要一开始就做 batching。

---

## 两个文档的衔接

- M01 结尾：模型能输出 logits，但还没有 cache；下一站 M02。
- M02 结尾：能生成文本，但每个请求一份连续 cache；用显存数字引出 M03 的分页。
- M02 的 `ContiguousKVCache` 要保留到 M04 作为对拍 oracle，不要删。
- M01/M02 都只依赖 M00 的 `SamplingParams`；不要提前引入 SchedulerOutput。

## 交付前检查

- [ ] 两个文件都存在，行数 320–480；
- [ ] 每个 vLLM 引用都有文件:行号或明确的符号；
- [ ] 每个实现任务都有验收命令；
- [ ] 至少 5 个常见坑；
- [ ] 有“教练流程”和“自测题”；
- [ ] 没有建议安装 vLLM，没有建议通读 giant file。
