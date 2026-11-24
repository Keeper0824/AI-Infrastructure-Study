# M09 · AsyncLLM 与 OpenAI 兼容服务

> **模块目标**：把同步的 `LLMEngine` 包成 `AsyncLLM`（每请求队列 + 后台 output loop + abort），
> 再套一层 FastAPI，做出 `/v1/completions`、`/v1/chat/completions`、`/v1/models`、`/health`。
> **预计工时**：4–5 小时　**依赖**：M08　**测试**：CPU（路由/SSE）+ GPU（端到端）
>
> **产出**：`curl -N` 能看到 SSE chunk 并以 `[DONE]` 结束；两个并发请求不串流；
> 客户端中途断开后引擎里的请求被 abort、KV 块被释放。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
pip install -e ".[server]"                 # fastapi / uvicorn / httpx
python -m nanovllm.entrypoints.openai_api --model models/Qwen3-0.6B --port 8000
# 另一个终端
curl -s localhost:8000/v1/models
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","prompt":"你好","max_tokens":16,"stream":true}'
python -m pytest tests/test_m09_serving.py -q          # 不需要 GPU：全用 fake engine
python -m pytest tests/test_m09_serving.py -q -m gpu   # 真模型端到端
```

以及 4 个能脱口而出的答案：为什么阻塞的 `step()` 不能直接在 async 路由里调？每请求队列
怎么隔离慢客户端？SSE 为什么必须要空行？abort 之后要清理哪几处状态？

---

## 1. 为什么需要这个模块（10 分钟）

M08 结束时你有一个**同步**引擎：`LLMEngine.step()` 阻塞、`generate()` 一次跑完所有请求、
没有任何网络边界。要变成“服务”，差的不是 HTTP 框架，而是三件事：

| 缺什么 | 后果 | 本模块怎么补 |
|---|---|---|
| 不阻塞事件循环的驱动方式 | 一个长请求卡死整个 server | 后台 task + `run_in_executor(step)` |
| 每请求的输出通道 | 所有请求共用一个返回值，没法流式 | `request_id -> asyncio.Queue` |
| 生命周期管理 | 客户端断了引擎还在算，KV 泄漏 | `abort()` 同时清前端状态和 core |

**边界**：M09 **不重写** detokenize、不重写采样、不碰 Scheduler。它只做“把
`EngineCoreOutputs` 变成 `RequestOutput` 再变成 JSON”这一段。M08 的 `OutputProcessor`
是唯一复用点；如果你在这里又写了一遍 detokenize，那一定是走错了。

进程边界先记住：M09 是 **in-process**（引擎和 FastAPI 同一个进程），M11 才把 EngineCore
挪进子进程。两边的分界线是 `EngineCoreRequest`（前端→core）和 `EngineCoreOutputs`
（core→前端）这两个可序列化结构，别的都不许跨界。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`。行号已 `grep -n` 校验（2026-10-29）。
> **注意**：这个版本的 OpenAI 入口已经拆成 `completion/`、`chat_completion/`、`models/`
> 三个子包，**没有** `serving_completion.py` / `serving_chat.py` / `protocol.py` 这三个
> 单文件——按旧路径去找会扑空。

### 2.1 必读（按顺序，约 350 行）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/engine/async_llm.py:84` | `AsyncLLM` | 字段：`engine_core` / `output_processor` / `output_handler`（一个 task） | 全部 `from_*` 构造器 |
| 2 | `vllm/v1/engine/async_llm.py:388` | `add_request()` | 建 `RequestState`、拿 queue、最后才 `engine_core.add_request` | 流式输入 / watermark 分支 |
| 3 | `vllm/v1/engine/async_llm.py:698` | `generate()` | **异步生成器**：取 queue → yield → `finished` 就停；异常里 abort | `STREAM_FINISHED` 细节 |
| 4 | `vllm/v1/engine/async_llm.py:825` | `_run_output_handler()` | 后台 task：`get_output_async` → `process_outputs` → 入队 → `reqs_to_abort` | 分块/chunk_size、metrics |
| 5 | `vllm/v1/engine/async_llm.py:911` | `abort()` | **两处都要清**：`output_processor.abort_requests` + `engine_core.abort_requests_async` | |
| 6 | `vllm/v1/engine/output_processor.py:681` | `process_outputs()` | M08 的产物；注意“有 queue 就 put，没 queue 就 return list” | stats/tracing |
| 7 | `vllm/v1/engine/output_processor.py:552` | `abort_requests()` | 前端侧状态怎么删 | |
| 8 | `vllm/v1/engine/core_client.py:353` | `InprocClient` | 注释说得很清楚：`get_output()` 就是**step 一次 core** | MPClient（M11 再看） |
| 9 | `vllm/v1/engine/__init__.py:109` | `EngineCoreRequest` | 前端→core 只传这几个字段 | mm/lora/kv_hints |
| 10 | `vllm/entrypoints/launchers/api_server/routers.py:12` | `register_api_routers()` | 路由是“注册”出来的，不是一个巨型 app | 其他 20 个 router |
| 11 | `vllm/entrypoints/openai/completion/api_router.py:47` | `create_completion()` | 3 个分支：ErrorResponse / Response / `StreamingResponse(media_type="text/event-stream")` | 负载指标 header |
| 12 | `vllm/entrypoints/openai/completion/serving.py:109` | `OpenAIServingCompletion.create_completion()` | 请求 → `SamplingParams`（`:172 to_sampling_params`）→ `async for` | 批量/echo/logprobs 分支 |
| 13 | `vllm/entrypoints/openai/completion/serving.py:279` | `completion_stream_generator()` | `f"data: {json}\n\n"`，最后 `data: [DONE]\n\n` | usage 的最终块 |
| 14 | `vllm/entrypoints/openai/chat_completion/api_router.py:54` | `create_chat_completion()` | 和 completions 结构一致 | |
| 15 | `vllm/entrypoints/openai/chat_completion/serving.py:281` | `create_chat_completion()` | `:360 to_sampling_params`；template 在 renderer 里 | tool calling / vision |
| 16 | `vllm/renderers/hf.py:759` | `safe_apply_chat_template()` | 模板渲染的唯一入口（`add_generation_prompt` 在 `vllm/renderers/inkling.py:121` 可见默认 True） | 多模态 content parts |
| 17 | `vllm/entrypoints/openai/completion/protocol.py:48` | `CompletionRequest` | 字段与默认值；`:343 to_sampling_params()` | 结构化输出字段 |
| 18 | `vllm/entrypoints/openai/chat_completion/protocol.py:220` | `ChatCompletionRequest` | `messages` / `max_completion_tokens` | tools |
| 19 | `vllm/entrypoints/openai/models/api_router.py:21` | `show_available_models()` | `/v1/models` 长什么样 | |
| 20 | `vllm/entrypoints/serve/instrumentator/health.py:22` | `@router.get("/health")` | 健康检查只要 200 | |

### 2.2 选读

| 位置 | 符号 | 为什么值得看一眼 |
|---|---|---|
| `vllm/entrypoints/serve/utils/api_utils.py:51` | `with_cancellation()` | 它明说 `request.is_disconnected()` 在中间件下不可靠，改用双 task 竞速 |
| `vllm/entrypoints/launchers/app.py:19` | `build_app()` | app 工厂的形状（nano 的 `build_app` 就是它的 20 行版） |
| `vllm/v1/engine/llm_engine.py:319` | `LLMEngine.step()` | 没有 queue 时的消费方式（`process_outputs` 直接返回 list） |
| `vllm/entrypoints/serve/engine/protocol.py:135` | `UsageInfo` | `usage` 三个字段名 |

### 2.3 明确跳过

多进程 `MPClient`（M11）、DP `DPCoordinator`（`vllm/v1/engine/coordinator.py:23`）、
metrics/Prometheus、鉴权、CORS、`responses` API、tool calling、vision/audio、
`stream_options.include_usage` 的最后一个空 choices 块、sse_keep_alive、OpenAI 的全部
可选参数。**不要通读** `async_llm.py`（1366 行）：你只需要第 84、388、698、825、911
这五个位置；也不要通读 `entrypoints/`——路由只是一个装饰器加一个 `include_router`。

---

## 3. 核心概念

### 3.1 同步引擎 + 异步前端

`EngineCore.step()` 是阻塞的（GPU 前向 + 调度），而 FastAPI 的路由跑在事件循环里。
**在 `async def` 里直接调 `step()`，整个 server 会在这几百毫秒里失去响应**——包括别的
请求、包括 `/health`。

```python
loop = asyncio.get_running_loop()
outputs = await loop.run_in_executor(None, self.engine_core.step)   # 丢到线程池
```

`InprocClient` 的 docstring 把这件事讲透了：in-process 模式下，`get_output()` **就是**
“step 一次 core”。所以 `AsyncLLM` 的后台 task 既是“输出分发器”，也是**引擎的驱动器**：
没人调 step，引擎就不动。

替代方案是专用线程 + `call_soon_threadsafe`；nano 用 `run_in_executor` 就够（一次只有一个 step）。

### 3.2 每请求队列

```python
self.queues: dict[str, asyncio.Queue] = {}
# add_request 时：q = asyncio.Queue(maxsize=256); self.queues[request_id] = q
# output loop：q.put_nowait(ro)；generate 结束：q.put_nowait(None) 当作哨兵
```

为什么不用一个全局队列：全局队列必须靠 `ro.request_id` 再分发一次，而且**慢客户端会
堵住所有请求**。每请求队列把背压限制在单个请求上。

`maxsize` 的意义：无界队列在客户端不读时会吃光内存。nano 的策略是“满了就 abort 这个请求”
（`QueueFull` → `abort`），而不是 `await q.put()` 把整个 output loop 挂住——后者是最贵的坑。

### 3.3 后台 output loop

```python
async def _output_loop(self):
    loop = asyncio.get_running_loop()
    while not self._closed:
        await self._has_work.wait()                      # 没请求就睡，不要忙等
        outs = await loop.run_in_executor(None, self.engine_core.step)
        for ro in self.output_processor.process_outputs(outs):
            q = self.queues.get(ro.request_id)
            if q is None: continue
            try:
                q.put_nowait(ro)
                if ro.finished: q.put_nowait(None)
            except asyncio.QueueFull:                     # 慢客户端只牺牲自己
                await self.abort(ro.request_id)
        for rid in self.output_processor.reqs_to_abort:   # 前端判到 stop string
            self.engine_core.abort_requests([rid])
        if not self.engine_core.has_unfinished_requests():
            self._has_work.clear()                        # 回睡
```

三个细节：`_has_work` 由 `add_request` 置位（`asyncio.Event` 跨 task 安全）；
`process_outputs` 必须在**同一个 task** 里调用（`asyncio.Queue` 不是线程安全的）；
`step()` 返回空 `EngineCoreOutputs` 是正常的（chunked prefill 中间步），别当“引擎空了”。

### 3.4 SSE 格式

```text
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"cmpl-...","choices":[{"index":0,"text":"你","finish_reason":null}]}

data: {"id":"cmpl-...","choices":[{"index":0,"text":"好","finish_reason":"stop"}]}

data: [DONE]

```

规则：每条消息 `data: <一行 JSON>` + **一个空行**（`\n\n`）才算一条事件；客户端（openai
python sdk、curl、浏览器 EventSource）都是靠空行切分的，少了空行就永远等不到完整事件。
最后一条是字面量 `data: [DONE]\n\n`（不是 JSON）。`Content-Type` 必须是
`text/event-stream`，用 `StreamingResponse(..., media_type="text/event-stream")` 设。

JSON 用 `json.dumps(..., ensure_ascii=False)`：中文不转义，体积小一半，也更好 debug。

### 3.5 chat template

```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
```

`add_generation_prompt=True` 会补上 `<|im_start|>assistant\n`。**漏了它，模型会接着
“用户”的角色继续写**，看起来像答非所问。不要手拼 `"user: ...\nassistant:"`——每个模型
的模板不同（Qwen3 还有 `enable_thinking` 开关，默认 True 会先输出 ` thinking`；对拍时
两边要传一样的参数）。

vLLM 把这件事放进 `Renderer`（`vllm/renderers/hf.py:759`），nano 直接调 tokenizer 即可。

### 3.6 abort：三处状态，一处都不能少

| 要清的东西 | 不清会怎样 |
|---|---|
| `self.queues[request_id]` | 队列泄漏；`generate()` 永远等不到 `None` |
| `OutputProcessor.request_states[request_id]` | 下次同名请求 `add_request` 直接复用旧 detokenizer |
| `EngineCore` 里 Scheduler/KV 的请求 | 请求继续生成、block 不释放（显存泄漏） |

vLLM 的 `abort()` 就是两行：`output_processor.abort_requests(...)` +
`engine_core.abort_requests_async(...)`。**只做第一行是最常见的错误。**

触发 abort 的三个来源：客户端断开（`StreamingResponse` 被取消 → `generate()` 收到
`GeneratorExit`/`CancelledError`）、前端判到 stop string（`reqs_to_abort`）、显式
`POST` 里的取消逻辑（nano 不做）。

### 3.7 OpenAI 响应形状

```jsonc
// 非流式
{"id":"cmpl-<hex>","object":"text_completion","created":1730000000,"model":"qwen3-0.6b",
 "choices":[{"index":0,"text":"你好！","finish_reason":"stop"}],
 "usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}
// 流式（每个 chunk）
{"id":"cmpl-<hex>","object":"text_completion","created":...,"model":...,
 "choices":[{"index":0,"text":"你","finish_reason":null}]}
```

`finish_reason` 只在**最后一个** chunk 非 null；中间的 chunk 必须是 `null`。chat 的
`object` 是 `chat.completion` / `chat.completion.chunk`，文本在
`choices[0].message.content`（非流式）或 `choices[0].delta.content`（流式）。
`usage` 可以简化，但三个字段必须在（vLLM 见 `vllm/entrypoints/serve/engine/protocol.py:135`）。

### 3.8 in-process 与 MP 的边界

M09 里 `AsyncLLM` 直接持有 `LLMEngine`（`engine_core` 同进程），`step()` 在线程池里跑。
M11 会把 `engine_core` 换成 `MPClient`：`add_request` 变成序列化 + 管道写，
`step()` 变成 `get_output()` 阻塞读。**只要 M09 把“调 core”收敛到两个方法**
（`add_request` / `step`），M11 就只是换实现，HTTP 层一行都不用改。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `AsyncLLM`（1366 行，ZMQ/多进程/引擎循环） | `nanovllm/engine/async_llm.py` | 只留 queue + output loop + abort，约 120 行 |
| `RequestOutputCollector` | `asyncio.Queue` + `None` 哨兵 | 不需要 `get_nowait` 优化 |
| `InprocClient` | 直接持有 `LLMEngine` | 不引入 `EngineClient` 抽象（M11 再加） |
| `MPClient` / `DPCoordinator` | 删 | M11 讲 multiprocessing |
| `Renderer` / `safe_apply_chat_template` | `tokenizer.apply_chat_template` | 不做多模态 content parts |
| `register_api_routers`（20+ router） | 一个 `build_app()` 里 4 个路由 | |
| `OpenAIServingCompletion`（900 行） | `entrypoints/openai_api.py` 里 2 个函数 | 删 echo/logprobs/batch |
| `CompletionRequest.to_sampling_params` | 一个 8 行映射函数 | 只映射本课程支持的字段 |
| `with_cancellation` 装饰器 | `try/except (CancelledError, GeneratorExit)` | nano 单请求粒度足够 |
| `UsageInfo` | 一个 dict | 三个 int |

**本模块冻结的接口**（`00_architecture.md` §6 未列，此处为准）：

```python
# nanovllm/engine/async_llm.py
class AsyncLLM:
    def __init__(self, llm_engine: LLMEngine) -> None: ...
    async def add_request(self, request_id: str, prompt: str,
                          sampling_params: SamplingParams) -> None: ...
    async def generate(self, prompt: str, sampling_params: SamplingParams,
                       request_id: str | None = None) -> AsyncIterator[RequestOutput]: ...
    async def abort(self, request_id: str) -> None: ...

# nanovllm/entrypoints/openai_api.py
def build_app(async_llm: AsyncLLM, model_name: str) -> FastAPI: ...
```

`generate()` 是**异步生成器**（函数体里有 `yield`），标注写 `AsyncIterator[...]`。
`build_app` 不启动 server；`main()` 里 `uvicorn.run(app, host, port)`。

---

## 5. 实现任务

### T1 · AsyncLLM 的队列与后台 loop（60 分钟）

文件：`nanovllm/engine/async_llm.py`。字段：`engine_core` / `output_processor` /
`queues` / `_has_work` / `_task` / `_closed`。`_output_loop` 见 §3.3；`add_request`
顺序必须是“先注册 queue，再 `engine_core.add_request`”，否则第一批输出会丢。

验收：`python -m pytest tests/test_m09_serving.py -q -k "output_loop"`
坑：① 在 `__init__` 里 `create_task` 会绑定到“当前”loop，server 启动前没有 loop，改用
lifespan；② 忘了 `_has_work.set()` → 第一次请求永远不动；③ 忙等 `while True: step()`
会把 CPU 打满。

### T2 · `generate()` 异步生成器（40 分钟）

```python
async def generate(self, prompt, sampling_params, request_id=None):
    rid = request_id or f"cmpl-{uuid.uuid4().hex}"
    q = asyncio.Queue(maxsize=256); self.queues[rid] = q
    try:
        await self.add_request(rid, prompt, sampling_params)
        while True:
            item = await q.get()
            if item is None: break
            yield item
            if item.finished: break
    except (asyncio.CancelledError, GeneratorExit):
        await self.abort(rid); raise
    finally:
        self.queues.pop(rid, None)
```

验收：`python -m pytest tests/test_m09_serving.py -q -k "generate"`
坑：① `finished` 只能 yield 一次，且是最后一条；② `finally` 里 pop 队列，但**不要**在
正常结束路径上 abort（会白跑一次 abort）；③ 用 `asyncio.Queue()` 而不是 `queue.Queue`。

### T3 · abort 与清理（30 分钟）

`async def abort(self, request_id)`：`output_processor.abort_requests([rid])` +
`engine_core.abort_requests([rid])`，再给队列塞 `None` 让 `generate()` 退出。若请求不存在
就静默返回（幂等）。

验收：`python -m pytest tests/test_m09_serving.py -q -k "abort"`
坑：① 只删前端状态 → KV 块不释放；② `abort` 后再 `abort` 会 KeyError；③ 别在
`process_outputs` 的循环里同步 `await abort`（会和自己抢队列）。

### T4 · FastAPI app + lifespan（40 分钟）

```python
@asynccontextmanager
async def lifespan(app):
    async_llm.start()          # create_task(_output_loop)
    yield
    await async_llm.shutdown() # 置 _closed、取消 task、join

app = FastAPI(lifespan=lifespan)
@app.get("/health")
async def health(): return {"status": "ok"}
@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": model_name, "object": "model"}]}
```

验收：`python -m pytest tests/test_m09_serving.py -q -k "app or health or models"`
坑：① 在 `__init__` 里建 task；② 关闭时没取消 task，pytest 报 “Task was destroyed”；
③ `/v1/models` 少了 `object: list` 时 SDK 解析失败。

### T5 · completions：非流式 + 流式（60 分钟）

`POST /v1/completions`：`CompletionRequest` → `SamplingParams`（8 行映射）→ 非流式
`async for` 收完拼 JSON；流式 `StreamingResponse(sse_gen(), media_type="text/event-stream")`。
SSE 生成器见 §3.4，`finish_reason` 只在最后一块非 null。

验收：`python -m pytest tests/test_m09_serving.py -q -k "completion"`
坑：① 非流式直接 `await` 全部输出却忘了检查 `finished`（可能死等）；② 流式 chunk 的
`text` 用了累积值 → 客户端拼出重复文本（用 M08 的 `delta_text`）；③ 少了 `\n\n`。

### T6 · chat：走 template（30 分钟）

`POST /v1/chat/completions`：`apply_chat_template(messages, tokenize=False,
add_generation_prompt=True)` 得到 prompt，之后完全复用 T5 的生成逻辑；响应换成
`message.content` / `delta.content`，`object` 换成 `chat.completion[.chunk]`。

验收：`python -m pytest tests/test_m09_serving.py -q -k "chat"`
坑：① 漏 `add_generation_prompt=True` → 模型接着用户写；② 把 `messages` 直接 `str()`
喂给模型；③ Qwen3 的 `enable_thinking` 默认 True，输出会带 ` thinking`，对拍时要统一。

### T7 · SSE 格式与错误处理（30 分钟）

单独抽一个 `_sse(data: dict) -> str` 返回 `f"data: {json.dumps(data, ensure_ascii=False)}\n\n"`；
结束时 yield `"data: [DONE]\n\n"`。请求非法（`max_tokens<=0`、`messages` 为空）返回 400 +
`{"error": {...}}`，不要抛 500。

验收：`python -m pytest tests/test_m09_serving.py -q -k "sse"`
坑：① 用 `\n` 而不是 `\n\n`；② `[DONE]` 也包成 JSON（`"data: \"[DONE]\"\n\n"`）；③
错误响应在流式里要作为**一个 chunk** 发出去，而不是改 HTTP 状态码（头已经发出去了）。

### T8 · 并发与断开测试（40 分钟）

两个不同 prompt 并发 `async with client.stream(...)`，逐块比对不串；中途 `break` 掉一个
流，等 1 秒后检查 `engine.has_unfinished_requests() == False` 且 `block_pool.num_free_blocks`
回到初值。用 `httpx.AsyncClient(transport=ASGITransport(app=app))`，不必起 uvicorn。

验收：`python -m pytest tests/test_m09_serving.py -q -k "concurrent or disconnect"`
坑：① 用同一个 `request_id` 起两个请求 → 队列互相覆盖；② 断开测试没等引擎收尾就断言；
③ `ASGITransport` 不会真正断开 TCP，要显式 `aclose()` 生成器来模拟。

### T9 · README 里的 curl 示例（20 分钟）

把 §0 的三条 curl 抄进 README，并补一条 `-d '...stream":false'` 的对照；注明
`qwen3-0.6b` 是 `build_app(model_name=...)` 里传的名字。

验收：`bash scripts/m09_smoke.sh`（三条 curl + `grep -c '^data: '` ≥ 2）
坑：① 忘了 `-H 'Content-Type: application/json'` → FastAPI 返回 422；② `-N` 关掉 curl
缓冲，不加看不到逐块输出；③ 端口占用时脚本静默失败。

---

## 6. 测试与对拍

| 测什么 | 怎么测 | 断言 |
|---|---|---|
| 非流式 | `POST /v1/completions`，`stream=false` | 200；`choices[0].text` 非空；`usage` 三字段在 |
| 流式 | `stream=true`，逐行读 | ≥2 个 `data: ` 块；最后 `[DONE]`；中间 `finish_reason` 为 null |
| 并发 | 两个不同 prompt 同时流式 | 两个流的文本各自连贯，无交叉 |
| 断开 | 读到一半 `aclose()` | 引擎里请求消失；`num_free_blocks` 恢复 |
| chat | `messages=[{"role":"user","content":"你好"}]` | 输出不含 `"user"` 字样（模板生效） |
| 元信息 | `GET /health`、`GET /v1/models` | 200；`data[0].id == model_name` |
| 一致性 | 同参数 vs `LLM.generate(greedy)` | 文本**逐字相同** |
| 关闭 | lifespan 退出 | 无 “Task was destroyed” 警告 |

**测试替身**：CPU 测试用 `FakeEngine`（`step()` 吐预设 token、`has_unfinished_requests`
按脚本翻转），路由/SSE/abort 全可测；GPU 测试只测“一致性与流式拼接”。

---

## 7. 常见坑

1. **在 async 路由里直接调 `engine_core.step()`**：整个事件循环被占住，`/health` 都超时。
2. **每请求队列不清理**：`queues` 只增不减，长跑必 OOM；`finally` 里 pop。
3. **output loop 里 `await queue.put`**：一个不读的客户端能让全引擎停摆；用 `put_nowait`
   + 有界队列 + 溢出即 abort。
4. **SSE 少空行**：客户端永远收不到完整事件（这是最难 debug 的一个，抓包只看得到字节）。
5. **chat 模板漏 `add_generation_prompt`**：模型自问自答，或直接续写用户的话。
6. **abort 只清前端**：`OutputProcessor` 干净了，KV 块还挂着，跑几轮就 OOM。
7. **流式 chunk 的 `finish_reason` 不合法**：中途就给 `"stop"`，或最后给 `null`，SDK 会报错。
8. **关服务不 join**：`uvicorn` 退出后后台 task 还在 step，进程挂住或显存不释放。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m09_serving.py -q`（CPU，FakeEngine）全绿；
- [ ] `pytest tests/test_m09_serving.py -q -m gpu` 全绿；
- [ ] `curl -N` 能看到 ≥2 个 `data:` 块并以 `[DONE]` 结束；
- [ ] 两个并发流不串；中途断开的请求在引擎里被 abort 且块被释放；
- [ ] `GET /health` 与 `GET /v1/models` 正常；chat 走模板；
- [ ] 与 `LLM.generate()` 同参数（greedy）输出逐字一致；
- [ ] `PROGRESS.md` 已更新，commit：`M09: AsyncLLM + OpenAI-compatible server`。

---

## 9. 交接：给 M10 / M11 的 agent

```text
nanovllm/engine/async_llm.py        新增 AsyncLLM（queue / output loop / generate / abort）
nanovllm/entrypoints/openai_api.py  新增 build_app() + 4 个路由 + SSE + 两个 main()
tests/test_m09_serving.py           新增（FakeEngine 的 CPU 测试 + -m gpu 端到端）
scripts/m09_smoke.sh                新增（3 条 curl）
README.md                           追加 curl 示例
```

- **给 M10**：`AsyncLLM._output_loop` 里 `run_in_executor(step)` 的耗时就是端到端延迟的
  天花板；benchmark 时把“排队时间”和“step 时间”分开记。
- **给 M11**：把 `self.engine_core` 换成 `MPClient`，只改 `add_request` / `step` /
  `abort_requests` 三个调用点；`EngineCoreRequest` 必须可序列化（现在是 dataclass，够用）。
- **给 M12**：`abort` 的三处清理、SSE 的 `finish_reason` 规则，是 review 时最值得检查的两处。

已知限制（写进 `PROGRESS.md`）：单 worker、无鉴权、无 CORS、`usage` 不含 cache 统计、
不做 `n>1`、不做 `stream_options.include_usage`、不支持 tool calling。

---

## 10. 自测题

1. 为什么不能在 async 路由里直接调 `EngineCore.step()`？两种正确的做法是什么？
2. 每请求队列相比全局队列，除了“分发简单”，还解决了什么问题？队列满了怎么办？
3. SSE 的格式是什么？为什么空行不能少？`[DONE]` 是 JSON 吗？
4. chat template 的 `add_generation_prompt=True` 为什么不能省？
5. 客户端断开后要清理哪三处状态？漏了哪一处会显存泄漏？
6. in-process 与 MP 的边界在哪两个数据结构上？M11 要改哪些调用点？

---

## 11. 教练流程（给 agent 用）

| 时间 | 做什么 | 禁止 |
|---|---|---|
| 20min | 讲 §3.1–3.3：为什么必须线程池 + 每请求队列 | 直接贴 FastAPI 代码 |
| 30min | 读 `generate()` 与 `_run_output_handler()` 两段 | 读整个 `async_llm.py` |
| 60min | 用户写 T1–T3，用 FakeEngine 跑通“无 HTTP”的异步生成 | 先写路由 |
| 60min | 用户写 T4–T7：先非流式，再 SSE，最后 chat | 三个一起写 |
| 40min | T8：并发 + 断开；抓 SSE 原始字节确认空行 | 只用 `requests` 测 |
| 30min | curl 验收 + README + `PROGRESS.md` | 跳过断开测试 |
| 15min | 自测题 + commit | 不跑测试就收工 |

**顺序提醒**：先把 `AsyncLLM` 单独跑通（`async for` 直接拿 `RequestOutput`，不经过 HTTP），
再接 FastAPI。HTTP 出问题时，先确认“绕开 HTTP 还对不对”，能省一半时间。
