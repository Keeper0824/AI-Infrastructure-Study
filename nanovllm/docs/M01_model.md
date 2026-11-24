# M01 · Qwen3 模型前向与 safetensors 权重加载

> **模块目标**：用纯 PyTorch 写出 Qwen3-0.6B 的 forward 和权重加载，和 HuggingFace
> `Qwen3ForCausalLM` 对拍 logits / greedy token。
>
> **预计工时**：3–4 小时　**依赖**：M00（只用 `SamplingParams` 的字段，前向不接 `Request`）
> **测试**：CPU 单测 + GPU 对拍
>
> **产出**：`python scripts/check_model_parity.py --model models/Qwen3-0.6B` 退出码 0。

---
## 0. 本模块结束时你会有什么

```bash
python scripts/check_model_parity.py --model models/Qwen3-0.6B \
  --prompt "The capital of France is" --max-new-tokens 16
# logits_max_abs_diff = 0.0xx    greedy_16_match = True    [ok] parity
```

以及 5 个能脱口而出的答案：head_dim 为什么不能推、q_norm 在 RoPE 前后、16 Q→8 KV 怎么映射、tied embedding、两次 residual。

**先决条件**（conda 环境 `vllm_learn`；不要装 vLLM）：

```bash
conda activate vllm_learn
python -c "import transformers, safetensors; print(transformers.__version__, safetensors.__version__)"
# vllm_learn 里应已是 5.10.1 / 0.8.0，不要盲目 -U 升级

# 只有缺 modelscope 时才装
python -m pip install modelscope

modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
python -c "import json; c=json.load(open('models/Qwen3-0.6B/config.json')); print(c['num_hidden_layers'], c['hidden_size'], c['num_attention_heads'], c['head_dim'])"
# 必须打印 28 1024 16 128
```

---
## 1. 为什么需要这个模块（10 分钟）

M02–M06 的所有对拍最终都要落回“logits 和 HF 一样”。这一层错了，后面就是在给错误的
oracle 做测试。vLLM 把模型定义、权重加载（`AutoWeightsLoader` + fused `qkv_proj`）、
可替换的 `CustomOp` 拆得很开；nano 只保留**计算图 + 老实加载器**：不 fused、不 TP/PP/
量化/LoRA、RMSNorm 就是一段 PyTorch、attention 先用 `F.scaled_dot_product_attention`。
本模块刻意不做采样、KV cache、batching、流式输出，只交付
`forward(input_ids, positions) -> logits [T, vocab]`。

---
## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`。行号于 2026-10-29 用
> `grep -n "<symbol>"` 校验；漂了就用符号重定位。阅读顺序：Attention / DecoderLayer
> 结构 → Qwen2MLP → RMSNorm 公式。RoPE 不要读 vLLM 的一千行实现。

### 2.1 必读（按顺序）

| # | 位置 · 符号 | 看什么 | 可跳过 |
|---|---|---|---|
| 1 | `vllm/model_executor/models/qwen3.py:65 · Qwen3Attention` / `vllm/model_executor/models/qwen3.py:156 · forward()` | `__init__` 用 `head_dim` 算 q/k/v 输出维度；`q_norm`/`k_norm` 是 `RMSNorm(head_dim)`；forward 里 `qkv.split`、q/k reshape、**先 norm 再 RoPE**、`self.attn`、`o_proj` | 量化、TP、prefix、`attn_type`、`self.attn` 的后端选择 |
| 2 | `vllm/model_executor/models/qwen3.py:176 · Qwen3DecoderLayer` / `vllm/model_executor/models/qwen3.py:231 · forward()` | 两个 norm 的位置；pre-norm + 两个 residual；vLLM 把 residual 融合进 norm，返回 `(hidden, residual)` | 融合细节、PP |
| 3 | `vllm/model_executor/models/qwen3.py:269 · Qwen3Model` | 只加了一个 `decoder_layer_type`；骨架在 qwen2.py 的 `Qwen2Model` | 其余 |
| 4 | `vllm/model_executor/models/qwen2.py:81 · Qwen2MLP` / `vllm/model_executor/models/qwen2.py:111 · forward()` | `gate_up_proj` + `SiluAndMul` + `down_proj`；三个矩阵；`gate_up -> act -> down`，无 bias | 量化、TP |
| 5 | `vllm/model_executor/models/qwen2.py:324 · Qwen2Model` / `vllm/model_executor/models/qwen2.py:402 · forward()` | `embed_tokens` / `layers` / `norm`；embedding → 逐层 → final norm 四步骨架 | PP、mrope、aux hidden、`intermediate_tensors` |
| 6 | `vllm/model_executor/models/qwen3.py:280 · Qwen3ForCausalLM` | `self.model` + `self.lm_head`；tie 时 `lm_head.tie_weights(embed_tokens)` | PP、LoRA、Eagle |
| 7 | `vllm/model_executor/models/qwen3.py:327 · forward()` / `vllm/model_executor/models/qwen3.py:339 · compute_logits()` | 只把 hidden 交给 `self.model`；logits 在 `compute_logits` 里算——M06 `execute_model`/`sample_tokens` 切分的雏形 | `LogitsProcessor` 内部 |
| 8 | `vllm/model_executor/models/qwen3.py:346 · load_weights()` | 只是委托给 `AutoWeightsLoader(self)` | `AutoWeightsLoader` 内部 |
| 9 | `vllm/model_executor/models/qwen3.py:284 · packed_modules_mapping` | `qkv_proj <- q/k/v`、`gate_up_proj <- gate/up`；**nano 不做融合** | Eagle / PP 的 key |
| 10 | `vllm/model_executor/layers/layernorm.py:37 · RMSNorm` / `vllm/model_executor/layers/layernorm.py:74 · forward_native()` / `vllm/ir/ops/layernorm.py:10 · rms_norm()` | 委托给 IR op；**公式在 IR**：转 float32 → 平方均值 → `rsqrt(var+eps)` → 乘 weight → 转回 | CUDA / fused 分支、其余 IR |
| 11 | `vllm/model_executor/layers/rotary_embedding/__init__.py:32 · get_rope()` | 只确认它按 `rope_theta` / `rope_scaling` / `head_size` 造 RoPE；nano 只支持默认 RoPE | YaRN / NTK / Phi3 / mrope 分支 |

### 2.2 建议读（选读，20 分钟以内）

- `qwen3.py:65-155`：只看 `q_size = num_heads * head_dim`、`kv_size = num_kv_heads * head_dim`、
  `self.rotary_emb = get_rope(...)` 三处。
- `qwen2.py:324-402`：只看 `embed_tokens` / `layers` / `norm` 的构造，不读 PP 分支。
- 若装了 `transformers`，对照 `modeling_qwen3.py` 的 `Qwen3Attention.forward`；只对符号，不记 HF 行号。

### 2.3 明确跳过

`AutoWeightsLoader` 内部、`packed_modules_mapping` 的加载算法、`LogitsProcessor`、
`ParallelLMHead`、`SupportsLoRA`/`SupportsEagle`、量化、TP/PP、mrope、`CustomOp`
dispatch、所有 RoPE scaling 变体。

读完 §2 你应该能回答：vLLM 的 `qkv_proj` 输出怎么 split？`q_norm` 的输入 shape 是
`[T, 1024]` 还是 `[T, 16, 128]`？`compute_logits()` 为什么独立？

---
## 3. 核心概念

先记四个数：`hidden=1024`、`Q heads=16`、`KV heads=8`、`head_dim=128`
（config 显式值）。于是 `q_dim=2048`、`kv_dim=1024`。

### 3.1 RMSNorm

```text
y = x / sqrt(mean(x^2, dim=-1) + eps) * weight      # eps=1e-6，在 sqrt 里面
```

和 LayerNorm 的区别：**不减均值、没有 bias**。数值上用 float32 算方差和 rsqrt，
乘 weight 后 cast 回输入 dtype；vLLM 的 `vllm/ir/ops/layernorm.py:10` 就是
`x.float()` → `pow(2).mean` → `rsqrt` → `to(weight.dtype) * weight` → `to(orig_dtype)`。
`RMSNorm` 只依赖最后一维，因此同一个类可服务 hidden 维（1024）和 per-head 维（128）。

### 3.2 RoPE

```text
inv_freq[i] = 1 / theta^(2i/d),  i=0..d/2-1
cos[t] = [cos(t*freq), cos(t*freq)]      # 前后两半相同
rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
q_rot = q*cos + rotate_half(q)*sin       # k 同理，v 不旋转
```

四个要点：只旋转 q/k；`positions` 是**绝对位置**；`rotate_half` 是前后半交换，
`[1,2,3,4] -> [-3,-4,1,2]`；cos/sin 可预计算成 float32 表，但进 attention 前必须
cast 到 q 的 dtype，否则类型提升会让结果对不上 HF。`theta=1e6`，表大小约 `40960 × 128 × 4 × 2 ≈ 40 MB`。

### 3.3 GQA

```text
q head:  0  1   2  3  ...  14 15
kv head:  0  0   1  1  ...   7  7      # h -> h // 2
```

`repeat_interleave(2, dim=1)` 把 `[1,8,L,128]` 变成 `[1,16,L,128]`，head `h` 用
kv `h // 2`。M01 用物理复制，代码最短；M04 的 paged attention 用 `h // group_size`
索引，省带宽。

### 3.4 QK-Norm

Qwen3 在 attention 前对**每个 head 的 128 维**做 RMSNorm：

```text
q: [T,16,128] --q_norm--> [T,16,128] --RoPE-->
k: [T, 8,128] --k_norm--> [T, 8,128] --RoPE-->
```

权重 `q_norm.weight` / `k_norm.weight` shape `[128]`，不是 `[1024]`。顺序是
**norm 在 RoPE 之前**（`qwen3.py:162-170`：view 成 head → norm → `rotary_emb`）。
漏掉它短 prompt 还能出词，但 logits 会和 HF 差一个数量级。

### 3.5 SwiGLU

```text
mlp(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )     # 1024→3072→3072→1024
```

三个 Linear 都无 bias；`silu(x)=x*sigmoid(x)`，用 `F.silu`。不要写成
`silu(gate+up)` 或 `silu(gate)+up`。

### 3.6 tied embeddings

`tie_word_embeddings=true` 时 `lm_head.weight` 与 `embed_tokens.weight` 是同一块
`[151936,1024]` 矩阵，Qwen3-0.6B 的 checkpoint 里**通常没有** `lm_head.weight`。
不能“缺了就跳过”（会随机初始化，输出噪声）；加载完 embedding 后把 `lm_head.weight`
指向它，并断言 `data_ptr` 相等。tie 掉可省约 `151936×1024×2 B ≈ 311 MB`。

### 3.7 pre-norm residual

```text
h = h + attn( input_layernorm(h) )
h = h + mlp( post_attention_layernorm(h) )      # 层间；最后再做一次 model.norm(h)
```

vLLM 为融合 kernel 把 residual 作为第二个返回值在层间传（`qwen3.py:231` 的
`(hidden, residual)` 协议）；nano 用 HF 的显式加法，数学等价、调试友好。**不要**
照抄融合协议，否则第一层就会少加一次。

---
## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| fused `qkv_proj` / `gate_up_proj` | `q_proj`/`k_proj`/`v_proj`、`gate_proj`/`up_proj` | 删融合，权重名与 HF 1:1 |
| `q_norm`/`k_norm`（`RMSNorm(head_dim)`） | 同名同 shape | 保留 |
| `get_rope()` + 各种 scaling | 一个 `RotaryEmbedding`，只支持默认 RoPE | `rope_scaling != None` 直接报错 |
| `Attention` 层 + backend + KV cache | `F.scaled_dot_product_attention`，无 cache | 简化；M02 加 cache，M04 换后端 |
| PP / mrope / aux hidden | 一个 `nn.ModuleList` 循环 | 删分支 |
| `ParallelLMHead` + `LogitsProcessor` | `nn.Linear(hidden, vocab, bias=False)` | 简化；保留 `compute_logits` |
| `AutoWeightsLoader` + packed mapping | 按 HF 名逐个 copy 的 loader | 简化，见 T7 |
| `RMSNorm(CustomOp)` + IR op | 纯 PyTorch `RMSNorm` | 公式照 IR op |
| TP / PP / 量化 / LoRA / Eagle | 无 | 单卡 |

**本模块新增的接口（非冻结，不进 `docs/00_architecture.md` §6）**：

```python
# nanovllm/models/qwen3.py
@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int; intermediate_size: int; num_hidden_layers: int
    num_attention_heads: int; num_key_value_heads: int; head_dim: int
    rms_norm_eps: float; rope_theta: float; vocab_size: int
    max_position_embeddings: int; tie_word_embeddings: bool; rope_scaling: dict | None = None
    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Qwen3Config": ...
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float): ...
    def forward(self, x): ...
class RotaryEmbedding(nn.Module):
    def forward(self, positions, q, k) -> tuple[torch.Tensor, torch.Tensor]: ...
class Qwen3MLP(nn.Module): ...
class Qwen3Attention(nn.Module):
    def forward(self, hidden_states, positions): ...
class Qwen3DecoderLayer(nn.Module): ...
class Qwen3Model(nn.Module):
    def forward(self, input_ids, positions) -> torch.Tensor: ...   # [T, hidden]
class Qwen3ForCausalLM(nn.Module):
    def forward(self, input_ids, positions) -> torch.Tensor: ...   # [T, vocab]
    def compute_logits(self, hidden_states) -> torch.Tensor: ...
    def load_weights(self, model_dir: str) -> list[str]: ...
```

约定：① **没有 batch 维**，`input_ids` 是 `[T]`，logits 是 `[T, vocab]`；
② `positions` 是 `torch.long`，长度等于 `T`；③ `load_weights` 返回已加载的名字
（vLLM 返回 missing keys，nano 反过来），方便断言 310/311 个张量；
④ `Qwen3Config` 是 models 层私有配置，**不**替代 §6.1 的 `ModelConfig`，本模块不建
`nanovllm/config.py`。

---
## 5. 实现任务

> 按顺序做；每个任务先写“会失败的测试”，再写实现。不要一次改 4 个文件。

### T1 · config 与模型骨架（25 分钟）

文件：`nanovllm/models/qwen3.py`、`nanovllm/models/__init__.py`。
`from_pretrained` 读 `config.json`；断言 `hidden_act == "silu"`、`attention_bias is False`、
`rope_scaling is None`；`head_dim` 必须来自 config，缺失才回退 `hidden // heads`。
验收：

```bash
python -c "from nanovllm.models.qwen3 import Qwen3Config; c=Qwen3Config.from_pretrained('models/Qwen3-0.6B'); assert (c.num_hidden_layers,c.hidden_size,c.head_dim)==(28,1024,128); print(c)"
```

坑：`torch_dtype` 是字符串 `"bfloat16"`；不要用 `hidden_size // 16` 覆盖 `head_dim`；
`Qwen3Config` 不要读 tokenizer 的文件。

### T2 · RMSNorm（20 分钟）

文件：`nanovllm/models/qwen3.py`。按 §3.1 实现；先写失败测试：

```python
x, w = torch.randn(4, 8), torch.randn(8)
norm = RMSNorm(8, 1e-6); norm.weight.data.copy_(w)
expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)
```

验收：`python -m pytest tests/test_m01_model.py -q -k "rms_norm"`。
坑：`keepdim=True`；`eps` 在 sqrt 内；bf16 输入必须 `x.float()` 再算方差。

### T3 · RoPE（30 分钟）

文件：`nanovllm/models/qwen3.py`。`RotaryEmbedding(head_dim, max_position, theta)`
预计算 `[max_position, head_dim]` 的 cos/sin buffer；`forward(positions, q, k)`
返回新张量，不改入参。验收：

```bash
python -m pytest tests/test_m01_model.py -q -k rope
```

至少两个测试：`rotate_half([1,2,3,4]) == [-3,-4,1,2]`；旋转后 `q·k` 只依赖
`pos_k - pos_q`（构造 `q=k`，平移两个位置比较）。
坑：写成 `x[..., ::2]` 是相邻交换；表在 CPU 而 q 在 CUDA；`positions` 用 int64
索引前不要转 float。

### T4 · MLP（20 分钟）

文件：`nanovllm/models/qwen3.py`。`gate_proj/up_proj: 1024→3072`，`down_proj: 3072→1024`，
`bias=False`，`return down(F.silu(gate(x)) * up(x))`。
验收：`python -m pytest tests/test_m01_model.py -q -k mlp`（和 HF `layers[0].mlp`
比输出，`atol=2e-2`）。坑：`silu(gate)*up` 顺序；`intermediate_size=3072` 不是
`4*hidden`；三个 Linear 都无 bias。

### T5 · attention（无 cache，40 分钟）

文件：`nanovllm/models/qwen3.py`。

```text
q,k,v 投影 → q.view(T,16,128), k/v.view(T,8,128)
q_norm/k_norm → RoPE → k/v repeat_interleave(2) → [1,16,T,128]
out = sdpa(q,k,v,is_causal=True) → [T,2048] → o_proj → [T,1024]
```

验收：`python -m pytest tests/test_m01_model.py -q -k attention`。测试：写一个
双层 for 循环的 `naive_attention` 和 SDPA 比；再和 HF `layers[0].self_attn` 比。
坑：`q_proj` 输出 2048 不是 1024，`o_proj` 输入也是 2048；q/k 必须 reshape 成
`[T,heads,128]` 再 norm；`repeat_interleave` 的 dim 是 1。

### T6 · decoder layer / model / lm_head（40 分钟）

文件：`nanovllm/models/qwen3.py`。decoder 按 §3.7 两次 residual；`Qwen3Model` 是
embedding → 28 层 → `self.norm`；`Qwen3ForCausalLM` 是 `self.model` + `self.lm_head`，
`compute_logits` 独立。验收：

```bash
python -m pytest tests/test_m01_model.py -q -k "model or logits"
```

测试：`forward` 输出 `[T,151936]`；`compute_logits(h) == lm_head(h)`；和 HF 全模型
最后一个位置比。坑：第二次 residual 用错变量；漏 final norm；`lm_head` 加了 bias；
在 `__init__` 里就 tie 会害 `named_parameters()` 去重（T7）。

### T7 · safetensors loader（35 分钟）

文件：`nanovllm/models/weight_loader.py`。要求：有 `model.safetensors.index.json`
就按 `weight_map` 读所有 shard，否则读单文件 `model.safetensors`；只支持 safetensors；
按 HF 名 1:1 加载（模块树命名成 `model.layers.{i}.self_attn.q_proj.weight`）；
只允许缺 `lm_head.weight`；顺序是 **build → `.to(dtype/device)` → load（末尾 tie）→
断言 `data_ptr` 相等**。

```python
state = {}
for shard in shards:
    with safe_open(f"{model_dir}/{shard}", framework="pt", device="cpu") as f:
        state.update({n: f.get_tensor(n) for n in f.keys()})
expected = {n for n, _ in self.named_parameters()}
assert expected - state.keys() <= {"lm_head.weight"} and not (state.keys() - expected)
for name, p in self.named_parameters():
    if name in state: p.data.copy_(state[name])
if config.tie_word_embeddings: self.lm_head.weight = self.model.embed_tokens.weight
```

验收：

```bash
python -m pytest tests/test_m01_model.py -q -k load
python -c "
from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
m = Qwen3ForCausalLM(Qwen3Config.from_pretrained('models/Qwen3-0.6B'))
print(len(m.load_weights('models/Qwen3-0.6B')), len(m.state_dict()))
assert len(m.state_dict()) == 311 and m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()"
```

坑：只读第一个 shard；`copy_` 前不查 shape（会静默广播）；tie 后 `named_parameters()`
去重，别把 `lm_head.weight` 误报成 missing。

### T8 · parity 脚本（30 分钟）

文件：`scripts/check_model_parity.py`。

```bash
python scripts/check_model_parity.py --model models/Qwen3-0.6B \
  --prompt "The capital of France is" --max-new-tokens 16 --atol 2e-2 --rtol 1e-2
```

四件事：① HF 显式 `torch_dtype=torch.bfloat16, attn_implementation="eager"`、
`.cuda().eval()`；② 比 `logits[0,-1]` 与 `nano_logits[-1]`；③ 无 cache 的朴素 greedy
16 步与 HF `generate(do_sample=False)` 逐 token 一致；④ hook 比 embedding / `layers[0]`
/ final norm / logits，超阈值 `sys.exit(1)`；hook 输出是 tuple 时取 `out[0]`，HF 的
batch 维先 `squeeze(0)`，nano 不用。验收：退出码 0 且打印 `greedy_16_match = True`。
坑：HF 默认 sdpa；bf16 比较前 `.float()`；nano 的 `logits[-1]` 不是 `logits[:, -1]`。

### T9 · 测试文件（25 分钟）

文件：`tests/test_m01_model.py`。CPU 测试覆盖 RMSNorm/RoPE/MLP 的公式，参数化
`head_dim=128`、`T=1`、`T=17`；GPU 测试标 `@pytest.mark.gpu`，用
`pytest.importorskip("transformers")` / `("safetensors")`，模型目录不存在则 `skip`；
固定 `torch.manual_seed(0)`、`torch.no_grad()`。验收：

```bash
python -m pytest tests/test_m01_model.py -q
python -m pytest tests/test_m01_model.py -q -m gpu
```

坑：CPU 测试不要 import CUDA；GPU 测试不要依赖执行顺序；模型构造用 module-scoped fixture。

---
## 6. 测试与对拍

**oracle**：

```python
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
ids = torch.tensor([tok("The capital of France is").input_ids], device="cuda")
with torch.no_grad():
    hf_logits = model(ids).logits          # [1,T,V]
    nano_logits = nano(ids[0], torch.arange(ids.shape[1], device="cuda"))  # [T,V]
```

**三层断言**：

| 层次 | 比较 | 阈值 | 失败先查 |
|---|---|---|---|
| 单元 | 公式 vs 实现 | `atol=1e-6`（float32） | keepdim、公式 |
| 模块 | HF 单层 vs nano 单层 | `atol=2e-2, rtol=1e-2` | head_dim、q_norm 顺序 |
| 端到端 | `hf_logits[0,-1]` vs `nano_logits[-1]` | 同上 | loader、tie、final norm |

`2e-2 / 1e-2` 是 bf16 经验值，不是理论保证。判断标准：**某一层从 1e-2 跳到 1e-1，
那里就是 bug**，不要用“bf16 就这样”糊过去。

**greedy 16**：每步用当前完整 `ids` 调 `nano`，取 `logits[-1].argmax()`，追加后再算
（O(T²)，16 token 无所谓；遇到 eos 就停，和 HF 一致）；和 HF `generate(ids,
do_sample=False, max_new_tokens=16, pad_token_id=tok.eos_token_id)[0, T:]` 逐 token 比。

**失败现象 → 原因**：

| 现象 | 原因 |
|---|---|
| 所有 logits 一样 / 输出重复 | loader 没加载、`lm_head` 随机、tie 没做 |
| 第一层 hidden 就差 0.1 | `head_dim=64`、q_norm 用了 hidden 维、RoPE 顺序错 |
| 短 prompt 正常长 prompt 崩 | `rope_theta` 没读（用了 10000）、positions 非绝对 |
| NaN | RMSNorm 漏 eps、权重未初始化、bf16 溢出 |
| 只有最后 token 差 | 正常 bf16 噪声，看最大绝对差是否 < 2e-2 |

**纪律**：不用 `torch.equal`；两边都 `.float()` 再比；greedy 差一个 token 时先比 logits 和 top-5。

---
## 7. 常见坑

1. **`head_dim` 用 `hidden_size // num_heads`**：1024/16=64，但 config 明写 128；
   q_proj 应是 `[2048, 1024]`，o_proj 输入 2048。用 64 时 shape 都“合理”，数值全错。
2. **q_norm/k_norm 作用在整个 hidden 上**：必须是 `[T, heads, 128]` 的最后一维；
   权重 `[128]`。错误写法不报 shape 错，只报数值错。
3. **q_norm 加在 RoPE 之后**：顺序是 `view → q_norm → RoPE`；反了短序列差异小、
   长序列差异大，很难查。对照 `qwen3.py:156-174`。
4. **`rotate_half` 写成相邻交换**：`x[..., ::2]` 是偶数位；正解是
   `cat(-x[..., d/2:], x[..., :d/2])`。用 `[1,2,3,4] -> [-3,-4,1,2]` 单测。
5. **cos/sin 的 dtype/device**：表在 float32/CPU，`table[positions]` 后忘了
   `.to(q.dtype)`，类型提升把 attention 变成 float32，和 HF 差一截。
6. **只读一个 safetensors 文件**：必须处理 `index.json`，否则换 1.7B/4B 静默少加载。
   验收断言 `len(state) in (310, 311)`。
7. **`lm_head.weight` 缺失就随机初始化**：tie 的 checkpoint 没有这个 key；忘了 tie
   会得到“语法像英语但内容随机”的输出。加载后断言 `data_ptr`。
8. **bf16 用 `torch.equal`**：同一份代码两次都可能差 1 ulp；用 `assert_close` 并写明
   atol/rtol。另外别忘了 `model.eval()` / `torch.no_grad()`，否则 28 层 activation
   全保留会 OOM。

---
## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m01_model.py -q` 与 `-q -m gpu` 全绿；
- [ ] `python scripts/check_model_parity.py --model models/Qwen3-0.6B` 退出码 0；
- [ ] `logits_max_abs_diff < 2e-2`，greedy 16 token 与 HF 完全一致；
- [ ] 逐层对比打印 embedding / layer0 / final norm / logits，并能解释第一处超阈值的阶段；
- [ ] 能不看代码回答 §10 的 5 个问题；
- [ ] `PROGRESS.md` 已更新（完成内容、已知偏差、M02 入口）；
- [ ] commit：`M01: Qwen3 forward + safetensors loader`。

---
## 9. 交接：给 M02 的 agent

**新增文件**：`nanovllm/models/__init__.py`、`nanovllm/models/qwen3.py`、
`nanovllm/models/weight_loader.py`、`scripts/check_model_parity.py`、
`tests/test_m01_model.py`。

**公开 API（M02 扩展时不破坏）**：

```python
Qwen3Config.from_pretrained(model_dir); Qwen3ForCausalLM(config)
model.load_weights(model_dir) -> list[str]
model(input_ids[T], positions[T]) -> [T, vocab]
model.compute_logits(hidden[T, hidden]) -> [T, vocab]
```

**已知限制**：无 KV cache（每步重算整段，O(T²)）；无 batch 维；只支持默认 RoPE、
bf16/float32、单卡；`is_causal=True` 只对 q/k 等长的 prefill 正确，M02 的 decode 要显式处理。

**M02 入口**：给 `Qwen3Attention.forward` / `Qwen3Model.forward` 加可选 `kv_cache`，
新增 `nanovllm/models/kv_cache.py`；新参数默认 `None`，`None` 时走本模块无 cache 路径。

---
## 10. 自测题

1. 为什么 `head_dim` 不能从 `hidden_size // num_heads` 推？算一遍 q_proj / o_proj 的 shape。
2. q_norm/k_norm 在 RoPE 前还是后？反了之后哪类输入最先暴露问题？
3. 16 个 Q head 怎么映射到 8 个 KV head？`repeat_interleave(2,dim=1)` 后 head 0/1/2/3
   分别对应哪个 KV head？
4. tied embedding 在加载时怎么处理？为什么不能“缺了就跳过”？
5. pre-norm residual 的两次加法分别在哪？final norm 在哪？

---
## 11. 教练流程（给 agent 用）

| 时间 | 做什么 | 检查点 |
|---|---|---|
| 20 min | 讲 §3 的四个数和 7 个概念，让用户复述公式 | 能手算 `rotate_half([1,2,3,4])` |
| 30 min | 带读 §2.1 的 1–4 条，只读 forward | 能指出 q_norm 在 RoPE 前 |
| 40 min | 用户写 T2/T3/T4，先写失败测试 | `pytest -k "rms_norm or rope or mlp"` 绿 |
| 60 min | 用户写 T5/T6，先和 HF 单层对拍再拼整模型 | 单层 attention 差 < 2e-2 |
| 30 min | 用户写 T7；卡住时提示“先数 key 的数量” | 310/311 个张量、`data_ptr` 相等 |
| 40 min | T8 parity + 逐层 debug，让用户定位第一处偏差 | parity 退出码 0 |
| 15 min | 自测题 + 更新 `PROGRESS.md` + commit | 5 题能答 |

**禁止**：一上来贴完整 `Qwen3ForCausalLM`（先让用户写 RMSNorm/RoPE，用 HF 单层对拍）；
通读 `qwen2.py` / RoPE 的一千行实现；用 `torch.equal` 做 bf16 对拍；在 T7 之前开始 T8
（loader 没通，parity 的失败信息全是噪声）。
