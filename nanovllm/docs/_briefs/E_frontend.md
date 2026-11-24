# 文档作者 brief E：M08 采样/输出 + M09 服务化

## 通用要求

工作区：`/home/dictator/ai_infra/nanovllm`

先读：

1. `docs/00_architecture.md`（§1–§6）
2. `docs/00_facts.md`（环境、行号、写作约定）
3. `docs/M00_contracts.md`（风格）
4. `docs/_template.md`
5. `ROADMAP.md`
6. `docs/_briefs/D_engine.md` 的 M06 部分（OutputProcessor / RequestOutput）

只写：

- `docs/M08_sampling_output.md`
- `docs/M09_serving.md`

不要改代码/其他文档。每个文件 350–480 行。中文。

---

## M08：Sampling、Logprobs、Detokenizer、Streaming

### 目标

把 M06 的最小 greedy 采样扩展成完整采样与输出管线：

- temperature / top-k / top-p / min-p；
- repetition / presence / frequency penalty；
- 每请求 seed + `torch.Generator`；
- logprobs（top-n + 采样 token 的 rank）；
- 增量 detokenize（不破坏 UTF-8）；
- stop strings / stop token ids / max_tokens；
- `RequestOutput` / `CompletionOutput`；
- 流式 `stream()`。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/sample/sampler.py:22` | `Sampler` | 类注释里的 9 步顺序 |
| `:73-150` | `Sampler.forward()` | raw logits / processed logits / logprobs |
| `:230-245` | `apply_temperature()` / `greedy_sample()` | temperature<eps 走 greedy |
| `:247-305` | `Sampler.sample()` | 先 greedy，再 temperature，再 top-k/p，再 `torch.where` |
| `:312-360` | `gather_logprobs()` | top-n + sampled token + rank |
| `vllm/v1/sample/metadata.py:15` | `SamplingMetadata` | 每个字段的来源 |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` | top-k/p 的实现；nano 用 PyTorch |
| `vllm/v1/sample/ops/penalties.py` | `apply_all_penalties()` | repetition/frequency/presence |
| `vllm/v1/sample/ops/logprobs.py` | `batched_count_greater_than` | rank 计算 |
| `vllm/v1/worker/gpu_input_batch.py:851` | `_make_sampling_metadata()` | 从 batch 建 metadata |
| `vllm/v1/engine/detokenizer.py:31` | `IncrementalDetokenizer` | 接口 |
| `:69-178` | `BaseIncrementalDetokenizer` | stop strings / `stop_buffer_length` |
| `:179-261` | `FastIncrementalDetokenizer` | `tokenizers.DecodeStream`；nano 可跳过 |
| `:262-330` | `SlowIncrementalDetokenizer` | 全量 decode + 增量；nano 参考这个 |
| `vllm/v1/engine/output_processor.py:136` | `RequestState` | prompt/output/status/detokenizer |
| `:616-740` | `add_request()` / `process_outputs()` | 主循环 |
| `vllm/v1/engine/logprobs.py` | `LogprobsProcessor` | 累积 logprobs |
| `vllm/v1/core/sched/utils.py:98` | `check_stop()` | token 级停止 |
| `vllm/sampling_params.py` | `SamplingParams` | 字段和默认值 |

**明确跳过**：`tokenizers.DecodeStream`、thinking budget、bad words、allowed token ids、
structured output、spec decode、n>1 并行采样、beam search、guided decoding。

### 必须讲清的概念

1. **采样顺序**：raw logits → penalties → temperature → top-k → top-p → min-p →
   softmax → multinomial。顺序不同结果不同；文档里要写死。
2. **greedy vs random**：`temperature < 1e-5` 走 argmax；否则随机。per-request 混合 batch
   要用 `torch.where` 合并，不能整 batch 一个模式。
3. **top-k**：取前 k 个 logits，其余 `-inf`；`k=-1` 或 `k>=vocab` 不限制。
4. **top-p**：按概率降序累加，保留累计 <= p 的最小集合；至少保留 1 个。
5. **min-p**：保留 `prob >= min_p * max_prob` 的 token；可选。
6. **penalties**：
   - repetition：对已出现 token 的 logit 除以 penalty（正负处理不同）；
   - frequency：按出现次数减；
   - presence：出现过就减一个常数。
7. **per-request generator**：`torch.Generator(device)` + `manual_seed(seed)`；
   没有 seed 用全局 RNG。测试要可复现。
8. **logprobs**：用 **raw logits**（penalty/temperature 之前）算 top-n，和 vLLM 一致；
   采样 token 的 logprob 从 processed logits 算，rank 从 raw logprobs 算。文档要说明
   这个区别，否则对拍对不上。
9. **增量 detokenize**：保存 `output_text`；每次 `decode(all_ids)` 得到新全文，取
   `new_text[len(old_text):]`；处理 `�`（不完整 UTF-8）时先不急着输出，等下一个 token。
10. **stop strings**：在 `output_text` 里找；如果 `include_stop_str_in_output=False`，
    输出要截断；还要处理跨 token 的 stop string（buffer 最大 stop 长度）。
11. **流式**：`stream()` 每步 yield 有增量的 `RequestOutput`；`finished=True` 时终止。
12. **finish_reason**：`stop` / `length` / `abort`；stop string 由前端判定，token 级
    停止由 Scheduler 判定。

### 必须实现的接口

```python
# nanovllm/sample/metadata.py
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor          # [num_reqs]
    top_p: torch.Tensor
    top_k: torch.Tensor
    min_p: torch.Tensor
    all_greedy: bool
    all_random: bool
    generators: dict[int, torch.Generator]
    max_num_logprobs: int | None
    output_token_ids: list[list[int]]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

# nanovllm/sample/sampler.py
class Sampler:
    def forward(self, logits: torch.Tensor,
                metadata: SamplingMetadata) -> SamplerOutput: ...
    def sample(self, logits, metadata) -> torch.Tensor: ...

# nanovllm/engine/detokenizer.py
class IncrementalDetokenizer:
    def __init__(self, tokenizer, prompt_token_ids, sampling_params): ...
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None: ...
    def get_next_output_text(self, finished: bool, delta: bool = True) -> str: ...

# nanovllm/engine/processor.py
class OutputProcessor:
    def add_request(self, request: Request) -> None: ...
    def process_outputs(self, outputs: EngineCoreOutputs) -> list[RequestOutput]: ...
```

`SamplerOutput` / `LogprobsLists` 用 `nanovllm/outputs.py` 的定义；可以扩展，但不要
改 `ModelRunnerOutput` 的字段。

### 实现任务

T1 SamplingMetadata；T2 penalties；T3 temperature/top-k/top-p/min-p；T4 seeded generator；
T5 logprobs；T6 IncrementalDetokenizer + stop strings；T7 OutputProcessor / RequestOutput；
T8 `LLM.stream()`；T9 测试。

### 测试与对拍

- greedy：与 HF `generate(do_sample=False)` 逐 token 一致；
- temperature=0.7 + 固定 seed：两次运行完全一致；
- top-k=1 等价 greedy；top-p=1.0 不裁剪；
- 分布测试：固定 logits，采样 10k 次，频率接近 softmax（卡方/粗检）；
- penalties：构造重复 token，确认 logit 被压；
- logprobs：`sum(exp(logprobs))≈1`；采样 token 的 logprob 与手动 `log_softmax` 一致；
- detokenizer：中文/emoji 多字节，增量输出拼起来等于全量 decode；
- stop strings：跨 token 的 stop、`include_stop_str_in_output` 两种；
- streaming：每步 delta 拼接等于最终 text；`finished` 只出现一次。

### 常见坑

- 先 temperature 再 penalties（顺序错）；
- top-p 对已经 `-inf` 的 token 排序，`cumsum` 出现 NaN；
- greedy 请求也走了 `multinomial`，导致不确定；
- per-request generator 用错 device / 没有 `manual_seed`；
- logprobs 用了 processed logits，和 vLLM 对不上；
- detokenizer 每步 `decode` 全量但直接输出，重复文本；
- 多字节字符被切成两半，输出 `�`；
- stop string 跨 token 没 buffer；
- EOS 被 `skip_special_tokens=False` 输出到文本里。

### 自测题

1. 采样顺序是什么？为什么 penalties 要在 temperature 之前？
2. top-p 的 cumulative 怎么算？为什么至少保留一个 token？
3. logprobs 应该用 raw 还是 processed logits？vLLM 为什么这样选？
4. 增量 detokenize 怎么处理不完整 UTF-8？
5. stop string 跨 token 怎么检测？
6. 为什么 greedy 和 random 混合 batch 要用 `torch.where`？

### 教练流程

概念 20min → 读 sampler 30min → 写 sampler 60min → 写 detokenizer/OutputProcessor 60min →
流式 30min → 测试 40min → 复盘 15min。先 greedy + detokenizer，再做 top-k/p；
不要一上来做 logprobs。

---

## M09：AsyncLLM + OpenAI 兼容服务

### 目标

- `AsyncLLM`：`async def generate()` 返回异步生成器；每个请求一个 `asyncio.Queue`；
  后台 output loop 把 `EngineCoreOutputs` 变成 `RequestOutput` 放进队列；
- FastAPI：
  - `POST /v1/completions`（stream / non-stream）
  - `POST /v1/chat/completions`（chat template）
  - `GET /v1/models`
  - `GET /health`
- 客户端断开时 abort；
- 并发请求 / 流式 SSE 测试。

### 必须覆盖的 vLLM 阅读

| 位置 | 符号 | 看什么 |
|---|---|---|
| `vllm/v1/engine/async_llm.py:84` | `AsyncLLM` | 字段：engine_core / output_processor / request queues |
| `:388-460` | `add_request()` | 建 RequestState、入队 |
| `:698-825` | `generate()` | 异步生成器、输出循环、abort |
| `:825-910` | `_run_output_handler()` | 后台任务：拿 EngineCoreOutputs → process_outputs → queue |
| `vllm/v1/engine/output_processor.py:681` | `process_outputs()` | 这里复用 M08 的 OutputProcessor |
| `vllm/v1/engine/input_processor.py:373` | `process_inputs()` | 分词、参数校验；nano 简化 |
| `vllm/entrypoints/openai/api_server.py` | `build_app` / routes | 只看路由注册和 lifespan |
| `vllm/entrypoints/openai/serving_completion.py` | `create_completion` | 请求 → SamplingParams |
| `vllm/entrypoints/openai/serving_chat.py` | `create_chat_completion` | chat template |
| `vllm/entrypoints/openai/protocol.py` | `CompletionRequest` / `ChatCompletionRequest` | 字段 |
| `vllm/v1/engine/core_client.py` | `InprocClient` / `MPClient` | 前端怎么拿 output；M09 用 in-process |
| `vllm/v1/engine/__init__.py:109` | `EngineCoreRequest` | 前端→core 的请求 |
| `vllm/v1/engine/coordinator.py` | `DPCoordinator` | 可以跳过（M11 再讲） |

**明确跳过**：多进程 client、DP、metrics/Prometheus、鉴权、CORS、OpenAI 的所有可选参数、
tool calling、vision、audio、responses API。

### 必须讲清的概念

1. **同步引擎 + 异步前端**：`EngineCore.step()` 是阻塞的；用
   `asyncio.get_running_loop().run_in_executor()` 或后台线程跑，避免阻塞事件循环。
2. **每请求队列**：`request_id -> asyncio.Queue[RequestOutput | None]`；`generate()`
   从队列取，`None` 表示结束。
3. **后台 output loop**：一个 task 循环 `engine_core.step()`（在线程里），把
   `EngineCoreOutputs` 交给 `OutputProcessor`，再 `put` 到对应队列。
4. **SSE 格式**：`data: {json}\n\n`；结束 `data: [DONE]\n\n`；`Content-Type:
   text/event-stream`。
5. **chat template**：`tokenizer.apply_chat_template(messages, tokenize=False,
   add_generation_prompt=True)`；不要手拼。
6. **abort**：客户端断开 → `request.is_disconnected()` → `engine_core.abort_requests([id])`；
   清理队列/RequestState。
7. **OpenAI 响应字段**：`id`、`object`、`created`、`model`、`choices`、`usage`；
   nano 可以简化 usage，但字段要存在。
8. **in-process vs MP**：M09 用 in-process；M11 再把 EngineCore 挪到子进程。

### 必须实现的接口

```python
# nanovllm/engine/async_llm.py
class AsyncLLM:
    def __init__(self, llm_engine: LLMEngine): ...
    async def add_request(self, request_id: str, prompt: str,
                          sampling_params: SamplingParams) -> None: ...
    async def generate(self, prompt: str, sampling_params: SamplingParams,
                       request_id: str | None = None) -> AsyncIterator[RequestOutput]: ...
    async def abort(self, request_id: str) -> None: ...

# nanovllm/entrypoints/openai_api.py
def build_app(async_llm: AsyncLLM, model_name: str) -> FastAPI: ...
```

FastAPI 路由：

```text
POST /v1/completions
POST /v1/chat/completions
GET  /v1/models
GET  /health
```

### 实现任务

T1 AsyncLLM 的队列/后台 loop；T2 generate 异步生成器；T3 abort/清理；
T4 FastAPI app + lifespan；T5 completions（stream/non-stream）；T6 chat（template）；
T7 SSE 格式；T8 并发/断开测试；T9 README 里的 curl 示例。

### 测试与对拍

- `curl` 非流式：返回 JSON，`choices[0].text` 正确；
- `curl -N` 流式：多个 `data:` chunk，最后 `[DONE]`；
- 2 个并发请求：输出不串；
- 客户端中途断开：引擎里请求被 abort，block 被释放；
- chat：`messages=[{"role":"user","content":"..."}]` 走 template；
- `GET /health` 200；`GET /v1/models` 有 model id；
- 与 `LLM.generate()` 同参数输出一致（ greedy ）。

### 常见坑

- 在 async 路由里直接调阻塞 `step()`，整个服务卡住；
- 每请求队列没有清理，内存泄漏；
- output loop 里 `await queue.put` 被慢客户端阻塞，拖垮整个引擎；
- SSE 少了空行 `\n\n`，客户端不解析；
- chat template 没有 `add_generation_prompt=True`；
- abort 只删了前端状态，没通知 EngineCore，KV 泄漏；
- 流式 chunk 的 `index`/`finish_reason` 不合法；
- 服务关闭时子任务/线程没有 join。

### 自测题

1. 为什么同步 `EngineCore.step()` 不能直接在 async 路由里调？
2. 每请求队列和全局队列的区别？慢客户端怎么隔离？
3. SSE 的格式是什么？为什么需要空行？
4. chat template 的 `add_generation_prompt` 为什么不能省？
5. 客户端断开后，哪些资源要清理？
6. in-process client 和 MP client 的边界在哪？

### 教练流程

概念 20min → 读 AsyncLLM.generate 30min → 写 AsyncLLM 60min → 写 FastAPI 60min →
流式/并发测试 40min → 复盘 15min。先做非流式 completions，再做 SSE，最后 chat。

---

## 两篇文档的衔接

- M08 的 `OutputProcessor` 是 M09 的核心复用点；M09 不要重写 detokenizer。
- M08 的 `RequestOutput` 字段要满足 M09 的 OpenAI 响应映射。
- M09 的 `AsyncLLM` 只包 in-process EngineCore；M11 再替换成 MPClient。
- M09 完成后，M10 做性能，M11 做进程边界，M12 收尾。

## 交付前检查

- [ ] 两个文件 350–480 行；
- [ ] M08 有采样顺序、top-p、logprobs、增量 detokenize 的细节；
- [ ] M09 有异步队列/后台 loop/SSE/abort 的细节；
- [ ] 两篇都有可执行的 curl / pytest 示例；
- [ ] 没有建议读完整 `async_llm.py`（1366 行）或 OpenAI serving 全集。
