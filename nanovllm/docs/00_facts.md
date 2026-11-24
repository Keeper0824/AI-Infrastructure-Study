# 00b · 事实表（写文档时先看这里）

> 这是给模块文档作者/agent 的“事实源”，避免每个模块重复查环境、重复引错行号。
> 如果和代码冲突，以代码为准，并回来修正本表。

## 1. 路径与版本

| 项 | 值 |
|---|---|
| 课程仓库 | `/home/dictator/ai_infra/nanovllm/` |
| vLLM 参照树 | `/home/dictator/ai_infra/target/vllm-0.32.0/` |
| vLLM 版本 | 目录名 `vllm-0.32.0`；worktree git 元数据缺失，无法用 tag 校验 |
| 引用格式 | `vllm/v1/request.py:60 · class Request` |
| 行号漂移 | 用 `grep -n "<symbol>"` 重定位；符号优先 |

## 2. 本机环境

课程统一使用 conda 环境 **`vllm_learn`**（不是 base）。

| 项 | 值 |
|---|---|
| conda env | `vllm_learn` |
| Python | 3.11.15（`/home/dictator/anaconda3/envs/vllm_learn/bin/python`） |
| torch | 2.11.0+cu130 |
| GPU | NVIDIA GeForce RTX 5080，16GB |
| Triton | 3.6.0 |
| 已装 | numpy 2.3.5、torch、triton、transformers 5.10.1、safetensors 0.8.0、fastapi、huggingface_hub |
| 默认缺 | **pytest**、modelscope |
| 不装 | vLLM 本体（要求 torch 2.13，会破坏现有环境） |

开工前：

```bash
conda activate vllm_learn
cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py
```

缺 pytest 时：

```bash
python -m pip install pytest
```

> base 环境（Python 3.13.5）有 pytest，但没有这套 torch/transformers；不要用它跑 M01+。

## 3. 模型：Qwen3-0.6B

`config.json` 的关键字段（来自 ModelScope `Qwen/Qwen3-0.6B`）：

| 字段 | 值 | 对实现的影响 |
|---|---|---|
| `num_hidden_layers` | 28 | 28 层 decoder |
| `hidden_size` | 1024 | |
| `intermediate_size` | 3072 | SwiGLU |
| `num_attention_heads` | 16 | Q heads |
| `num_key_value_heads` | 8 | GQA，group size = 2 |
| `head_dim` | 128 | `hidden_size / heads = 64` 但 config 显式给 128，**必须用 128** |
| `rms_norm_eps` | 1e-6 | |
| `rope_theta` | 1e6 | |
| `max_position_embeddings` | 40960 | |
| `vocab_size` | 151936 | |
| `tie_word_embeddings` | true | `lm_head.weight` 可能不存在，复用 embedding |
| `attention_bias` | false | QKV 无 bias |
| `hidden_act` | silu | SwiGLU |
| `torch_dtype` | bfloat16 | |

权重名（HF 格式）：

```text
model.embed_tokens.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.self_attn.q_norm.weight
model.layers.{i}.self_attn.k_norm.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.norm.weight
lm_head.weight   # tie_word_embeddings=true 时可能缺失
```

模型下载：

```bash
cd /home/dictator/ai_infra/nanovllm
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
```

HF 镜像备选：`HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B`。

## 4. 已验证的 vLLM 行号（2026-10-29 校验）

> 这些是写文档时可以直接引用的锚点。新增引用请自己 `grep` 验证。

### Request / SchedulerOutput

| 符号 | 位置 |
|---|---|
| `class Request` | `vllm/v1/request.py:60` |
| `append_output_token_ids()` | `vllm/v1/request.py:272` |
| `update_block_hashes()` | `vllm/v1/request.py:285` |
| `num_tokens_with_spec` | `vllm/v1/request.py:299` |
| `class RequestStatus` | `vllm/v1/request.py:370` |
| `class NewRequestData` | `vllm/v1/core/sched/output.py:41` |
| `class CachedRequestData` | `vllm/v1/core/sched/output.py:138` |
| `class SchedulerOutput` | `vllm/v1/core/sched/output.py:232` |
| `class ModelRunnerOutput` | `vllm/v1/outputs.py:267` |
| `class EngineCoreOutput` | `vllm/v1/engine/__init__.py:199` |
| `class EngineCoreOutputs` | `vllm/v1/engine/__init__.py:259` |
| `class FinishReason` | `vllm/v1/engine/__init__.py:48` |

### Scheduler / KV

| 符号 | 位置 |
|---|---|
| `class Scheduler` | `vllm/v1/core/sched/scheduler.py:84` |
| `schedule()` | `vllm/v1/core/sched/scheduler.py:591` |
| 主运行循环 | `vllm/v1/core/sched/scheduler.py:659` |
| waiting 循环 | `vllm/v1/core/sched/scheduler.py:870` 起 |
| `_preempt_request()` | `vllm/v1/core/sched/scheduler.py:1571` |
| `_update_after_schedule()` | `vllm/v1/core/sched/scheduler.py:1616` |
| `update_from_output()` | `vllm/v1/core/sched/scheduler.py:2026` |
| `_update_request_with_output()` | `vllm/v1/core/sched/scheduler.py:2493` |
| `add_request()` | `vllm/v1/core/sched/scheduler.py:2616` |
| `check_stop()` | `vllm/v1/core/sched/utils.py:98` |
| `class FCFSRequestQueue` | `vllm/v1/core/sched/request_queue.py:75` |
| `class KVCacheManager` | `vllm/v1/core/kv_cache_manager.py:131` |
| `get_computed_blocks()` | `vllm/v1/core/kv_cache_manager.py:264` |
| `allocate_slots()` | `vllm/v1/core/kv_cache_manager.py:371` |
| `free()` | `vllm/v1/core/kv_cache_manager.py:620` |
| `cache_blocks()` | `vllm/v1/core/kv_cache_manager.py:782` |
| `class BlockPool` | `vllm/v1/core/block_pool.py:136` |
| `get_cached_block()` | `vllm/v1/core/block_pool.py:198` |
| `cache_full_blocks()` | `vllm/v1/core/block_pool.py:226` |
| `get_new_blocks()` | `vllm/v1/core/block_pool.py:666` |
| `touch()` | `vllm/v1/core/block_pool.py:752` |
| `free_blocks()` | `vllm/v1/core/block_pool.py:774` |
| `class KVCacheBlock` | `vllm/v1/core/kv_cache_utils.py:176` |
| `NONE_HASH` | `vllm/v1/core/kv_cache_utils.py:118` |
| `hash_block_tokens()` | `vllm/v1/core/kv_cache_utils.py:695` |
| `get_request_block_hasher()` | `vllm/v1/core/kv_cache_utils.py:941` |

### Worker / Attention

| 符号 | 位置 |
|---|---|
| `class GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py:467` |
| `_update_states()` | `vllm/v1/worker/gpu_model_runner.py:1172` |
| `_prepare_inputs()` | `vllm/v1/worker/gpu_model_runner.py:1920` |
| `execute_model()` | `vllm/v1/worker/gpu_model_runner.py:3940` |
| `sample_tokens()` | `vllm/v1/worker/gpu_model_runner.py:4304` |
| `load_model()` | `vllm/v1/worker/gpu_model_runner.py:5028` |
| `profile_run()` | `vllm/v1/worker/gpu_model_runner.py:6133` |
| `capture_model()` | `vllm/v1/worker/gpu_model_runner.py:6482` |
| `initialize_kv_cache()` | `vllm/v1/worker/gpu_model_runner.py:7055` |
| `class CachedRequestState` | `vllm/v1/worker/gpu_input_batch.py:35` |
| `class InputBatch` | `vllm/v1/worker/gpu_input_batch.py:90` |
| `add_request()` | `vllm/v1/worker/gpu_input_batch.py:345` |
| `remove_request()` | `vllm/v1/worker/gpu_input_batch.py:523` |
| `class BlockTable` | `vllm/v1/worker/block_table.py:49` |
| `compute_slot_mapping()` | `vllm/v1/worker/block_table.py:153` |
| `class AttentionBackend` | `vllm/v1/attention/backend.py:59` |
| `class CommonAttentionMetadata` | `vllm/v1/attention/backend.py:387` |
| `class AttentionMetadataBuilder` | `vllm/v1/attention/backend.py:589` |
| `class TritonAttentionMetadata` | `vllm/v1/attention/backends/triton_attn.py:58` |
| `TritonAttentionMetadataBuilder.build()` | `vllm/v1/attention/backends/triton_attn.py:209` |
| `class TritonAttentionImpl` | `vllm/v1/attention/backends/triton_attn.py:385` |
| `kernel_unified_attention` | `vllm/v1/attention/ops/triton_unified_attention.py:179` |
| `unified_attention()` | `vllm/v1/attention/ops/triton_unified_attention.py:815` |
| `class Sampler` | `vllm/v1/sample/sampler.py:22` |
| `Sampler.forward()` | `vllm/v1/sample/sampler.py:73` |
| `Sampler.sample()` | `vllm/v1/sample/sampler.py:247` |
| `class SamplingMetadata` | `vllm/v1/sample/metadata.py:15` |
| `class RMSNorm` | `vllm/model_executor/layers/layernorm.py:37` |
| `class Qwen3Attention` | `vllm/model_executor/models/qwen3.py:65` |
| `Qwen3Attention.forward()` | `vllm/model_executor/models/qwen3.py:156` |
| `class Qwen3DecoderLayer` | `vllm/model_executor/models/qwen3.py:176` |
| `class Qwen3ForCausalLM` | `vllm/model_executor/models/qwen3.py:280` |
| `class Qwen2MLP` | `vllm/model_executor/models/qwen2.py:81` |
| `class Qwen2Model` | `vllm/model_executor/models/qwen2.py:324` |

### Engine / Frontend

| 符号 | 位置 |
|---|---|
| `class LLMEngine` | `vllm/v1/engine/llm_engine.py:54` |
| `add_request()` | `vllm/v1/engine/llm_engine.py:229` |
| `step()` | `vllm/v1/engine/llm_engine.py:319` |
| `class EngineCore` | `vllm/v1/engine/core.py:115` |
| `EngineCore.step()` | `vllm/v1/engine/core.py:657` |
| `class EngineCoreProc` | `vllm/v1/engine/core.py:1128` |
| `run_busy_loop()` | `vllm/v1/engine/core.py:1547` |
| `class AsyncLLM` | `vllm/v1/engine/async_llm.py:84` |
| `AsyncLLM.generate()` | `vllm/v1/engine/async_llm.py:698` |
| `class InputProcessor` | `vllm/v1/engine/input_processor.py:41` |
| `process_inputs()` | `vllm/v1/engine/input_processor.py:373` |
| `class OutputProcessor` | `vllm/v1/engine/output_processor.py:504` |
| `process_outputs()` | `vllm/v1/engine/output_processor.py:681` |
| `class IncrementalDetokenizer` | `vllm/v1/engine/detokenizer.py:31` |
| `class Executor` | `vllm/v1/executor/abstract.py:48` |
| `class UniProcExecutor` | `vllm/v1/executor/uniproc_executor.py:34` |
| `determine_available_memory()` | `vllm/v1/worker/gpu_worker.py:634` |
| `initialize_from_config()` | `vllm/v1/worker/gpu_worker.py:853` |

## 5. 写作约定

- 中文；代码标识符/文件路径保留英文。
- 每个模块文档 300–500 行，不要复制 `00_architecture.md` 的大段内容。
- 每个“读什么”都写清楚“看什么”和“可跳过什么”。
- 每个实现任务都有：文件、函数签名、验收命令、3 个坑。
- 不要给完整实现；给伪代码/接口/失败测试。
- 不要发明新的公共接口；如果必须加，明确写“本模块新增”，并说明为什么。
- 不要引用未验证的行号；用 `grep -n` 确认，或只写符号不写行号。
- 不要建议用户安装 vLLM 本体。
- 不要建议用户通读 `gpu_model_runner.py` / `scheduler.py` 全文。
