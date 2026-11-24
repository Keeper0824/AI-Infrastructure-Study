# 进度

> 每个模块结束后更新这个文件。下一个 agent 窗口会先读它，用来恢复上下文。
> 不要删旧记录；只追加新的“当前状态”。

## 当前状态

- **当前模块**：M01（已完成）
- **最近提交**：M00: contracts (request / scheduler output / block hash)（上一次提交）；M01 待提交
- **下一步**：M02 — 读 `docs/M02_naive_kv.md`，给 `Qwen3Attention.forward` / `Qwen3Model.forward` 加可选 `kv_cache`，新增 `nanovllm/models/kv_cache.py`

## 环境

- conda env: `vllm_learn`（不是 base）
- Python 3.11.15
- torch 2.11.0+cu130
- GPU: RTX 5080 16GB
- Triton 3.6.0
- transformers 5.10.1 / safetensors 0.8.0
- pytest 9.2.0（已安装）
- 模型：Qwen3-0.6B（已下载到 `models/Qwen3-0.6B`，单文件 `model.safetensors`，311 个 key）

## 开工前

```bash
conda activate vllm_learn
cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py
```

## 已完成

### M00 · 全景与契约层

- `Request` / `RequestStatus` / `SamplingParams`（`SamplingParams` 为骨架已有）
- `hash_block_tokens()` / `get_request_block_hasher()`
- `SchedulerOutput` / `NewRequestData` / `CachedRequestData` / `validate()`
- `python -m pytest tests -q` → **34 passed**
- 数据流：`Request`（可变）→ `SchedulerOutput`（本步指令）→ `ModelRunnerOutput`（worker 增量）→ `EngineCoreOutput`（回前端增量）；M00 只实现前两个 + 块哈希，后两个是空壳。
- T7 手推 trace（A prompt=6, B prompt=3, `max_tokens=2`, budget=4, block_size=4）：
  - step 1：new A，A:4，total 4，computed A=4
  - step 2：new B，cached A，A:2/B:2，total 4，computed A=6/B=2；A 采样出第一个 output token
  - step 3：cached A、B，A:1/B:1，total 2，computed A=7/B=3；A 在本步结束完成
  - step 4：cached B，B:1，total 1；`finished_req_ids={A}`
- 关键模型：`num_computed_tokens` 只加本步调度的 token；本步采样出的 token 要等下一步才算进 KV；请求完成后最后一个采样 token 不再计算。

### M01 · Qwen3 前向与 safetensors 加载

- 新增文件：`nanovllm/models/qwen3.py`（`Qwen3Config` / `RMSNorm` / `RotaryEmbedding` / `Qwen3MLP` / `Qwen3Attention` / `Qwen3DecoderLayer` / `Qwen3Model` / `Qwen3ForCausalLM`）、`nanovllm/models/weight_loader.py`、`tests/test_m01_model.py`、`scripts/check_model_parity.py`。
- 结构：`head_dim=128`；`q_norm` / `k_norm` 是 `RMSNorm(128)`，在 RoPE 之前；16 Q / 8 KV，`repeat_interleave(2, dim=1)`；两个 pre-norm residual；final norm；`lm_head` 无 bias。
- RoPE：cos/sin 表放在 `Qwen3Model` 层，28 层共用一张；表由 positions 索引后 cast 到 q 的 dtype。
- Loader：支持 `model.safetensors.index.json` 分片和单文件；按 HF 名 1:1 加载；允许缺 `lm_head.weight`；模型 `.to(bf16)` 后加载；末尾 tie 并断言 `data_ptr`。本机 checkpoint 是 311 个 key（含 `lm_head.weight`），`loaded=311`。
- 测试：`python -m pytest tests/test_m01_model.py -q` → **23 passed**；`-q -m gpu` → **2 passed**；`python -m pytest tests -q` → **57 passed**。
- parity：`python scripts/check_model_parity.py --model models/Qwen3-0.6B --prompt "The capital of France is" --max-new-tokens 16` → embedding / layer0 / final_norm / logits 全部 `0.00000`，`greedy_16_match=True`，退出码 0。
- T8 的 HF oracle 用 `attn_implementation="sdpa"`，不是文档里的 `eager`。原因：nano 也用 SDPA；bf16 下 HF eager 与 SDPA 的全模型 logits 差 0.39，float32 下差 1.7e-5。换成 SDPA 后逐层为 0.0，仍然验证权重、RoPE、GQA、norm、residual 和 lm_head。

## 已知偏差 / 待办

- 没有 spec decode 的实际调度；`spec_token_ids` 只保留字段，恒为空。
- `SchedulerOutput` 删除了 encoder / KV connector / PP / spec / async 字段。
- `preempted_req_ids` / `resumed_req_ids` 只保留字段，M05 才会实际使用。
- `Request` 未实现多模态 / LoRA / structured output / streaming session；`priority` 字段保留但调度器尚未使用。
- M00 代码清理（不阻塞 M01）：删除残留的 `TODO(M00-*)` / 注释掉的 `raise`；hasher 改用公开属性；`append_output_token_ids()` 对非法类型抛 `TypeError`；`validate()` 的 helper 和 assert 文案可简化。
- M01：attention 用 `F.scaled_dot_product_attention`，没有 KV cache、没有 batch 维，decode 每步重算整段（O(T²)）；只支持默认 RoPE（`rope_scaling=None`）；T8 oracle 用 HF SDPA 而不是 eager；`test_load_weights` 在 CPU 上加载 311 个权重，约数秒。

## 模块交接

### M00 → M01

- 新增/修改文件：`nanovllm/request.py`、`nanovllm/sched/output.py`、`nanovllm/utils/hashing.py`；测试未改。
- 测试命令：`python -m pytest tests -q` → **34 passed**
- 已知限制：M00 只定义契约，没有 Scheduler / Engine / Model；`ModelRunnerOutput` / `EngineCoreOutput` 仍是空壳；没有 spec decode。
- M01 入口：
  1. 读 `docs/M01_model.md`；
  2. 确认 `models/Qwen3-0.6B/config.json` 和权重文件存在；
  3. M01 会用到 `Request.prompt_token_ids` 和 `SamplingParams`；不需要 `SchedulerOutput`；
  4. 环境使用 `conda activate vllm_learn`（Python 3.11.15）。

### M01 → M02

- 新增/修改文件：`nanovllm/models/qwen3.py`、`nanovllm/models/weight_loader.py`、`tests/test_m01_model.py`、`scripts/check_model_parity.py`、`.gitignore`（`models/` → `/models/`，避免忽略源码目录）。
- 测试命令：`python -m pytest tests/test_m01_model.py -q` → **23 passed**；`python -m pytest tests -q` → **57 passed**。
- 公开 API：
  - `Qwen3Config.from_pretrained(model_dir)`
  - `Qwen3ForCausalLM(config)`
  - `model.load_weights(model_dir) -> list[str]`
  - `model(input_ids[T], positions[T]) -> [T, vocab]`
  - `model.compute_logits(hidden[T, hidden]) -> [T, vocab]`
- 已知限制：
  - `Qwen3Attention.forward(hidden_states, positions)` 没有 `kv_cache` 参数；
  - `Qwen3Model.forward(input_ids, positions)` 也没有 `kv_cache` 参数；
  - `is_causal=True` 只对 q/k 等长的 prefill 正确；M02 的 decode 是 q_len=1、k_len=T，PyTorch `is_causal=True` 的 top-left 对齐会只看到第一个 key，必须显式传 mask 或临时用 `is_causal=False` + mask；
  - 没有 batch 维；只支持默认 RoPE、bf16/float32、单卡。
- M02 入口：
  1. 读 `docs/M02_naive_kv.md`；
  2. 新增 `nanovllm/models/kv_cache.py`；
  3. 给 `Qwen3Attention.forward` / `Qwen3Model.forward` 加可选 `kv_cache`，默认 `None` 时保持 M01 的无 cache 路径；
  4. 先做连续 KV cache 的 correctness oracle，M03 再换 paged。
