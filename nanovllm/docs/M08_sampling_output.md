# M08 · 采样、Logprobs、Detokenizer 与流式输出

> **模块目标**：把 M06 的“最小 greedy + 全量 decode”扩成完整输出管线——采样参数、penalties、
> per-request 随机源、logprobs、增量 detokenize、stop 条件、`RequestOutput`、`stream()`。
> **预计工时**：3–4 小时　**依赖**：M06　**测试**：CPU（采样/detokenizer）+ GPU（端到端）
> **产出**：`pytest tests/test_m08_*.py -q` 全绿；`python scripts/m08_demo.py --stream` 能吐中文；
> 一张自己画的“采样九步顺序”图。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m08_sampling.py tests/test_m08_detokenizer.py -q  # CPU，不加载模型
python -m pytest tests/test_m08_output.py -q -m gpu                          # 真模型 Qwen3-0.6B
```

以及 5 个能脱口而出的答案（= §10 的前 5 题）。

---

## 1. 为什么需要这个模块（10 分钟）

M06 收尾时：`sample_tokens()` 里只有 `logits.argmax(-1)`，`OutputProcessor` 只有“每步
`decode(all_ids)` 然后把全文塞进 `RequestOutput`”。能跑，但有四个硬伤：

| 硬伤 | 现象 | 本模块怎么修 |
|---|---|---|
| 只有 greedy | `temperature=0.7` 被静默忽略 | `Sampler` 完整九步 |
| 没有增量 detokenize | 流式客户端收到重复全文 | `get_next_output_text(delta=True)` |
| 没有 stop string | `stop=["\n\n"]` 无效 | 前端判 stop + 通知 core abort |
| 没有 logprobs/finish_reason | M09 的 JSON 缺字段 | `LogprobsLists` + `FinishReason` |

**边界先记牢**：EngineCore/Scheduler 判 **token 级**停止（eos / stop_token_ids /
max_tokens / max_model_len）；前端判**文本级**停止（stop strings），判到后必须回头让 core
停。stop string 是文本概念，core 只有 token id；`max_tokens` 是计数概念，前端别越权。

接缝两处：`ModelRunner.sample_tokens()` 里把 `argmax` 换成 `self.sampler.forward(logits,
metadata)`；`OutputProcessor.process_outputs()` 是 M09 的**唯一**复用点。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`。下表行号已 `grep -n` 校验
> （2026-10-29）；漂了就用符号重定位，别猜。

### 2.1 必读（按顺序，约 400 行）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/sample/sampler.py:22` | `Sampler` | **类注释的 9 步顺序**，本模块的宪法 | bad_words / thinking budget |
| 2 | `vllm/v1/sample/sampler.py:73` | `Sampler.forward()` | raw logits 存快照 → fp32 → processors → sample → gather | `logprobs_mode` 分支只看默认 |
| 3 | `vllm/v1/sample/sampler.py:247` | `Sampler.sample()` | argmax → temperature → min_p → top-k/p → `torch.where` | `logprob_token_ids` 分支 |
| 4 | `vllm/v1/sample/sampler.py:230` | `apply_temperature()` | `temp < 1e-5` 的行换成 1.0，避免除零 | |
| 5 | `vllm/v1/sample/sampler.py:242` | `greedy_sample()` | 一行 `argmax(-1)`；CPU 分支可跳过 | `torch.ops._C` |
| 6 | `vllm/v1/sample/sampler.py:312` | `gather_logprobs()` | top-n + sampled 拼一起 + rank | `mark_unbacked` 几行 |
| 7 | `vllm/v1/sample/metadata.py:15` | `SamplingMetadata` | 每个字段来自哪里（对照 `_make_sampling_metadata`） | bad_words/spec/thinking 字段 |
| 8 | `vllm/v1/sample/ops/topk_topp_sampler.py:473` | `apply_top_k_top_p_pytorch()` | 升序 + `cumsum <= 1-p` + `[:, -1] = False` | 另外 5 个 backend |
| 9 | `vllm/v1/sample/ops/topk_topp_sampler.py:555` | `random_sample()` | 为什么不用 multinomial；`generators` 逐行加噪 | flashinfer 分支 |
| 10 | `vllm/v1/sample/ops/penalties.py:10` | `apply_all_penalties()` | 三个 penalty 的入参形状；`vocab_size` 当 pad | async 的 `-1` 补丁 |
| 11 | `vllm/model_executor/layers/utils.py:43` | `apply_penalties()` | repetition 走 custom op；freq/pres 减 `bincount*coef` | |
| 12 | `vllm/_custom_ops.py:370` | `apply_repetition_penalties_torch()` | **正负号**：`logit>0` 除以 p，否则乘 p | CUDA 版 |
| 13 | `vllm/v1/worker/gpu_input_batch.py:851` | `_make_sampling_metadata()` | `all_greedy`/`all_random`/`no_top_k`/`no_penalties` 从哪来 | |
| 14 | `vllm/v1/engine/detokenizer.py:69` | `BaseIncrementalDetokenizer` | `stop_buffer_length = max(len(stop)) - 1` | `min_tokens` 几行 |
| 15 | `vllm/v1/engine/detokenizer.py:96` | `update()` | 追加 token → `decode_next` → `check_stop_strings` | spec 多 token 的 `bisect` |
| 16 | `vllm/v1/engine/detokenizer.py:161` | `get_next_output_text()` | `finished` 时 buffer 归零；`delta` 用字符偏移 | |
| 17 | `vllm/v1/engine/detokenizer.py:262` | `SlowIncrementalDetokenizer` | 名字叫 Slow，其实仍是**增量** decode（prefix/read offset） | |
| 18 | `vllm/v1/engine/output_processor.py:681` | `process_outputs()` | 一个 for 循环：detokenize → stop → logprobs → 造输出 → 清理 | stats/tracing/mm |
| 19 | `vllm/v1/core/sched/utils.py:98` | `check_stop()` | eos / stop_token_ids / max_tokens / max_model_len 的**顺序** | min_tokens / repetition_detection |
| 20 | `vllm/sampling_params.py:268` | `SamplingParams` | 默认值（注意 vLLM 的 `top_k=0`、`temperature=1.0`） | 结构化输出 / watermark |

### 2.2 选读

| 位置 | 符号 | 为什么值得看一眼 |
|---|---|---|
| `vllm/v1/sample/ops/logprobs.py:11` | `batched_count_greater_than()` | rank 就是 `(x >= v).sum(-1)` |
| `vllm/v1/engine/logprobs.py:360` | `LogprobsProcessor.update_from_output()` | top-n 已含 sampled 时要去重 |
| `vllm/v1/sample/logits_processor/builtin.py:102` | `MinPLogitsProcessor.apply()` | min_p 官方实现，8 行 |
| `vllm/outputs.py:129` | `RequestOutput` / `CompletionOutput` | 字段名照抄，M09 直接映射 |
| `vllm/v1/outputs.py:205` | `SamplerOutput` | `[num_reqs, 1]` 采样结果 + logprobs |

### 2.3 明确跳过（第一遍看都别看）

`FastIncrementalDetokenizer` 与 `tokenizers.DecodeStream`、`spaces_between_special_tokens`、
thinking budget、bad words / allowed token ids、structured output、spec 的 bonus token、n>1 与
beam search、`processed_*` 两种 logprobs 模式、`stream_interval`、stats/tracing/多模态分支。
**不要通读** `output_processor.py`（983 行）或 `detokenizer.py`（373 行）。

---

## 3. 核心概念

### 3.1 采样顺序：写死，别改

```text
logits [R, V] fp32
 ① raw_logprobs = log_softmax(logits)          # 快照，只给 logprobs 用
 ② penalties：repetition → frequency → presence
 ③ temperature：logits.div_(temp)，temp<1e-5 的行按 1.0
 ④ min_p：丢掉 prob < min_p * max_prob 的 token
 ⑤ top_k：只留最大的 k 个，其余 -inf
 ⑥ top_p：留“累计概率达到 p 的最小前缀”
 ⑦ softmax → 按行采样（每请求一个 generator）
 ⑧ torch.where(temp < 1e-5, argmax, sampled)   # greedy 行用 argmax 覆盖
 ⑨ gather logprobs（用 ① 的快照）
```

三条“为什么”：**penalties 在前**——freq/pres 是对数空间减法、repetition 是除法，先除温度
等于让惩罚强度随温度缩放（`temperature=2` 时减半），vLLM 与 HF 都是先惩罚后温度。
**min_p 在 top_k/top_p 之前**——vLLM 把它归到 `argmax_invariant` processor，先于
`topk_topp_sampler` 执行；反过来 `max_prob` 会因 top-k 已打进 -inf 而变大，阈值跟着变
（brief 把 min-p 写在后面，以本节为准）。**where 在最后**：见 3.2。

### 3.2 greedy 与 random 混在一个 batch 里

`all_greedy` 直接 `argmax`（温度/top-k/p/采样全跳过）；`all_random` 连 argmax 都不算；
混合（两者都 False）**两条都算**，再 `torch.where(temp < 1e-5, greedy, random)`，省不掉。

混合时不能逐请求 if/else：① `if` 会引入 `.item()` 式的 GPU→CPU 同步；② 逐请求走不同
kernel，torch RNG 消耗顺序随 batch 组成变化，**同一个 seed 换 batch 就变结果**。
`torch.where` 逐元素选择，greedy 行白算一次随机，但 RNG 消耗确定：**分支换确定性**。

### 3.3 top-k / top-p / min-p 的精确公式

```python
# top-k（k=-1 或 k>=V 不裁）；注意 <，并列项保留，实际保留数可能 > k
kth = logits.topk(k, dim=-1).values[:, -1:]
logits = logits.masked_fill(logits < kth, -inf)
# top-p（p=1.0 整段跳过）：降序、含当前项的 cumsum
sl, idx = logits.sort(-1, descending=True)
probs = sl.softmax(-1); cum = probs.cumsum(-1)
drop = (cum - probs) >= p        # 「不含自己就够 p」→ 自己不要
drop[:, 0] = False               # 至少留最大的那个
logits = logits.scatter(-1, idx, sl.masked_fill(drop, -inf))   # 别忘了还原
# min-p（min_p=0 跳过）：此刻 logits 已除过温度
probs = logits.softmax(-1)
logits = logits.masked_fill(probs < probs.max(-1, keepdim=True).values * min_p, -inf)
```

vLLM 的 top-p 是**升序 + `cumsum <= 1-p` + `[:, -1] = False`**，与上式等价：升序里
`cum <= 1-p` 那段正是降序里“跨过 p 之前”那段，`[:, -1]`（升序最大项）= 降序第 0 项。

### 3.4 penalties：三个公式，两个集合

| penalty | 作用集合 | 公式（对数空间） |
|---|---|---|
| repetition | **prompt ∪ output** | `logit>0 ? logit/p : logit*p` |
| frequency | **只 output** | `logit -= freq * count(token)` |
| presence | **只 output** | `logit -= pres * (count>0)` |

易错点：**正负号**（负 logit 乘 p 更负；无脑 `logit/p` 会让 -5 变成 -4.17，等于奖励重复，
见 `_custom_ops.py:370`）；**repetition 吃 prompt**（`prompt_mask | output_mask`，HF 也吃整段
`input_ids`）；**freq/pres 只吃 output**（OpenAI 定义是 completion 部分）。

nano 不建 `[R, V]` 稠密 mask（V=151936，256 行 = 152MB），逐行只对出现过的 id 做 scatter：
`uniq, cnt = (prompt_ids + output_ids).unique(return_counts=True)`，在 `logits[r, uniq]` 上
where / 减法（vLLM 用稠密 mask 是因为有 CUDA kernel 扫整行）。

### 3.5 per-request generator

```python
gen = torch.Generator(device="cuda"); gen.manual_seed(sp.seed)   # add_request 时建一次
metadata.generators = {row: gen}      # key 是 batch 行号；无 seed 不放入
out[row] = torch.multinomial(probs[row], 1, generator=generators.get(row))
```

三条规则：**没 seed 不进 dict**（走全局 RNG；vLLM 的 `random_sample()` 也先整批加噪再逐行
覆盖）；**与 logits 同 device**（CPU generator 配 CUDA 张量会 mismatch）；**只有 seed 能跨
batch 复现**（全局 RNG 消耗取决于 batch 组成，固定 seed 的请求单跑与混在 8 路结果相同 = T4）。

vLLM 不用 `multinomial` 而用 Gumbel-max（`probs / Exponential(1)` 取 argmax），因为
`torch.multinomial` 会触发 CPU-GPU 同步。两者**随机流不同**：随机采样只能对分布。

### 3.6 logprobs：raw 还是 processed

vLLM V1 默认 `logprobs_mode="raw_logprobs"`（`sampler.py:64`）：top-n、采样 token 的
logprob、rank **三者都来自 `log_softmax(raw logits)`**（penalties/temperature 之前）。
`forward()` 注释原话：*"Use the original logits (before any penalties or temperature
scaling) for the top-k logprobs. This is different from the V0 sampler."*

用 processed 算：`temperature=0.7` 把分布拉平，logprob 偏大，`sum(exp(lp))` 不再等于 1。
**nano 跟 V1：全部用 raw**（早期笔记里“sampled 从 processed 算”是 V0 的旧行为）。口径：`-1` 全词表、
`0` 只返回 sampled、`None` 不要；顺序 `[sampled, top1, top2, ...]`；`cumulative_logprob` 累加。

### 3.7 增量 detokenize：全量 decode + 前缀差

```python
def update(self, new_token_ids, stop_terminated):
    self.token_ids.extend(new_token_ids)
    text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
    if text.endswith("\ufffd"):        # 半个汉字/emoji 先不吐
        text = text[:-1]
    self.output_text += text[len(self.output_text):]
```

三个坑对应三行：`skip_special_tokens=False` 时 EOS 会以 `<|im_end|>` 混进正文；一个汉字
可能占 3 个 token，`decode` 半个字符得 `�`，**只处理结尾**；`text[len(old):]` 的前提是每次
从**完整** `token_ids` 重 decode。代价 O(n²)，几十 token 无所谓（vLLM 用 prefix/read offset）。

### 3.8 stop strings：buffer 是 `max_len - 1`

`stop = ["\n\n", "###"]` → `stop_buffer_length = max(2, 3) - 1 = 2`。为什么 `-1`：长度 L
的 stop 在“还差最后一个字符”时最危险，留 L 个会把整个 stop 扣住；留 L-1 能在凑成之前把
安全字符吐出去，凑成那步又不会多吐。检测要从“可能跨块”处起 `find`（`detokenizer.py:321`）：

```python
idx = output_text.find(stop_str, max(0, len(output_text) - new_char_count - len(stop_str)))
```

命中后：`include_stop_str_in_output=True` 截到 stop **末尾**，`False` 截到**起点**；
`finish_reason="stop"`、`stop_reason=<字符串>`；core 还在跑就把 req_id 放进 `reqs_to_abort`，
由调用方交给 `engine_core.abort_requests()`。

### 3.9 finish_reason 与流式输出

| 值 | 谁判 | 触发 |
|---|---|---|
| `stop` | Scheduler（eos/stop_token_ids）或 OutputProcessor（stop string） | 正常结束 |
| `length` / `abort` | Scheduler / EngineCore | 截断 / 断开或前端判到 stop |

流式三条不变式（T8/T9 断言）：`finished=True` **只出现一次**且是最后一块；每步
`delta_text` 拼接 == 最终 `text`；stop 命中那步起 `text` 不再增长（除非 include）。
nano 的 `CompletionOutput` 同时给 `text`（累积）与 `delta_text`（增量）：vLLM 只有累积 text，
delta 由 OpenAI 层自己 diff；多存一个字段就省掉 M09 那份有状态的 diff。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `vllm/v1/sample/sampler.py:22 Sampler` | `nanovllm/sample/sampler.py` | 保留九步；删 bad_words/allowed ids/thinking/spec |
| `vllm/v1/sample/metadata.py:15 SamplingMetadata` | `nanovllm/sample/metadata.py` | 保留 12 字段 + `prompt_token_ids` |
| `apply_all_penalties` | `nanovllm/sample/ops.py` | 稀疏 scatter，不建稠密 mask |
| `SlowIncrementalDetokenizer` | `nanovllm/engine/detokenizer.py` | **全量 decode + 前缀差** |
| `FastIncrementalDetokenizer` | 删 | 依赖 `tokenizers.DecodeStream` |
| `RequestOutput`/`CompletionOutput` | `nanovllm/outputs.py` | 加 `delta_text`/`stop_reason`；删 metrics/lora |
| `SamplerOutput` | `nanovllm/outputs.py` | **新增** |
| `check_stop()` | `nanovllm/sched/utils.py`（M05） | 只留 eos/stop_ids/max_tokens/max_model_len |

**本模块新增/扩展的接口**（`00_architecture.md` §6 未列，此处为准）：

```python
# nanovllm/sample/metadata.py
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor; top_p: torch.Tensor; top_k: torch.Tensor   # [num_reqs] fp32
    min_p: torch.Tensor; all_greedy: bool; all_random: bool
    generators: dict[int, torch.Generator]   # key = batch 行号
    max_num_logprobs: int | None
    prompt_token_ids: torch.Tensor | None    # 新增：repetition 要吃 prompt
    output_token_ids: list[list[int]]
    frequency_penalties: torch.Tensor; presence_penalties: torch.Tensor; repetition_penalties: torch.Tensor

# nanovllm/outputs.py（扩展，不改 ModelRunnerOutput 的字段）
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor          # [num_reqs, 1]
    logprobs: LogprobsLists | None = None
# CompletionOutput 追加：delta_text: str = ""、stop_reason: int | str | None = None

# nanovllm/engine/processor.py
class OutputProcessor:
    def add_request(self, request: Request, prompt: str) -> None: ...  # 比 brief 多一个 prompt
    def process_outputs(self, outputs: EngineCoreOutputs) -> list[RequestOutput]: ...
```

`add_request` 多一个 `prompt`：`Request` 按 §6.2 只存 `prompt_token_ids`，而
`RequestOutput.prompt` 必须是原文（vLLM 的 `OutputProcessor.add_request` 也这么传）。

---

## 5. 实现任务

> 按顺序做；先跑失败测试再写实现，不要一次改三个文件。

### T1 · `SamplingMetadata`（20 分钟）

文件：`nanovllm/sample/metadata.py`，构建放 `InputBatch.make_sampling_metadata()`：把每行
`sampling_params` 抽成 fp32/int32 tensor → `.to(device)`；`all_greedy = all(temp < 1e-5)`；
`prompt_token_ids` 用 `[R, max_len]` padded int64（pad = `vocab_size`）。

验收：`python -m pytest tests/test_m08_sampling.py -q -k "metadata"`

坑：① 两者可同时为 False、不能同时为 True（加 assert）；② 必须 fp32（bf16 的 cumsum
精度不够）；③ `generators` 的 key 是行号，`remove_request` 后会 condense，须每次重建。

### T2 · penalties（35 分钟）

文件：`nanovllm/sample/sampler.py` 的 `apply_penalties()`。逐行取
`prompt_ids[padded != V] + output_ids[row]` → `unique(return_counts=True)` → repetition 用
`where(logits>0, /p, *p)`，frequency 减 `freq*cnt`，presence 减 `pres*1`。

验收：`python -m pytest tests/test_m08_sampling.py -q -k "penalty"`

坑：① repetition 用 prompt∪output、freq/pres 只用 output；② pad 值不过滤会越界；③ 原地改前确认 contiguous。

### T3 · temperature / top-k / top-p / min-p（45 分钟）

文件：`nanovllm/sample/sampler.py`，顺序固定为 §3.1 的 ③④⑤⑥，每步都要有整批短路：
`if (top_p == 1.0).all(): 跳过`、`if not all_greedy: 才除温度`。

验收：`python -m pytest tests/test_m08_sampling.py -q -k "top_k or top_p or temperature or min_p"`

坑：① nano 用 `top_k=-1` 表示不限制（vLLM 用 0），别把 0 当“留 0 个”；② 漏了
`drop[:, 0] = False` 时 p 很大会整行 -inf → NaN；③ 排序后必须 `scatter` 还原。

### T4 · seeded generator（20 分钟）

文件：`nanovllm/worker/input_batch.py`（`add_request` 建 generator）+ `sampler.py`（用）。
只在 `add_request` 时 `manual_seed` 一次。

验收：`python -m pytest tests/test_m08_sampling.py -q -k "seed or reproduce"`
（断言：单请求跑两次一致；单请求 vs 8 路 batch 一致）

坑：① device 必须和 logits 一致；② 每步重设 seed 会每步采到同一 token；③ 没 seed 的请求别塞 dict。

### T5 · logprobs（30 分钟）

文件：`nanovllm/sample/sampler.py` + `worker/model_runner.py`（D2H 成 list）。
`raw = logits.float().log_softmax(-1)`；`tok_lp = raw.gather(-1, sampled[..., None])`；
`rank = (raw >= tok_lp).sum(-1)`；topk 后拼成 `[sampled, top1..topn]`；`sample_tokens()`
里转 `LogprobsLists`（list，不是 tensor，因为要进 `ModelRunnerOutput`）。

验收：`python -m pytest tests/test_m08_sampling.py -q -k "logprob"`

坑：① 用 processed logits → 差一个温度因子；② `n=0` 时 topk 报错，先特判；③ `-1` 时每行长度 = vocab，别共享同一个 list。

### T6 · `IncrementalDetokenizer` + stop strings（50 分钟）

文件：`nanovllm/engine/detokenizer.py`。

```python
class IncrementalDetokenizer:
    def __init__(self, tokenizer, prompt_token_ids, sampling_params): ...
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None: ...
    def get_next_output_text(self, finished: bool, delta: bool = True) -> str: ...

# get_next_output_text 核心三行
buffer = 0 if finished else self.stop_buffer_length
end = len(self.output_text) - buffer
return self.output_text[self._last_offset:end] if delta else self.output_text[:end]
```

```bash
python -m pytest tests/test_m08_detokenizer.py -q
# 中文/emoji：每步 delta 拼接 == tokenizer.decode(all_ids)
# stop="\n\n"：命中那步 text 不含它；include_stop_str_in_output=True 时含
```

坑：① `stop_terminated=True` 时把最后一个 token 排除出 detokenize（`skipped_stop_token_id`），
否则 eos 被解成文本；② `_last_offset` 是**字符**偏移不是 byte；③ `update()` 返回字符串。

### T7 · `OutputProcessor` / `RequestOutput`（45 分钟）

文件：`nanovllm/engine/processor.py`。

```python
for out in outputs.outputs:
    st = self.request_states[out.request_id]
    stop_str = st.detokenizer.update(out.new_token_ids, out.finish_reason == FinishReason.STOP)
    finished = out.finish_reason is not None or stop_str is not None
    delta = st.detokenizer.get_next_output_text(finished, delta=True)
    text  = st.detokenizer.get_next_output_text(finished, delta=False)
    if stop_str and out.finish_reason is None: self.reqs_to_abort.append(out.request_id)
```

验收：`python -m pytest tests/test_m08_output.py -q -k "processor"`

坑：① 两次 `get_next_output_text` 必须用同一个 `finished`，否则 buffer 不一致；② stop 命中
要写 `finish_reason="stop"` + `stop_reason`，不能留 `None`；③ `reqs_to_abort` 每次开头清空。

### T8 · `LLM.stream()`（20 分钟）

文件：`nanovllm/entrypoints/llm.py`（接口见 §6.7）。先 add 全部请求，再
`while has_unfinished_requests(): for out in step(): yield out`。

验收：`python -m pytest tests/test_m08_output.py -q -k "stream" -m gpu`

坑：① `finished=True` 后别再 yield；② `step()` 空列表是正常的（chunked prefill 中间步）；
③ 生成器被 `break` 时引擎里还有请求，要 abort 或耗完迭代器。

### T9 · 测试（40 分钟）

`tests/test_m08_sampling.py`（纯 CPU，手工 logits）、`tests/test_m08_detokenizer.py`（真
tokenizer + 中文/emoji）、`tests/test_m08_output.py`（`-m gpu`）。CPU 测试不许 import 权重。
验收：`python -m pytest tests/test_m08_sampling.py tests/test_m08_detokenizer.py -q`
坑：① CPU 测试里 import 了 transformers 模型 → CI 无 GPU 直接挂；② 用假 tokenizer 测中文，
测不出 `�`；③ GPU 测试没打 `-m gpu`，在纯 CPU 机器上默认跑了。

---

## 6. 测试与对拍

| 测什么 | 怎么测 | 断言 |
|---|---|---|
| greedy 对拍 | nano greedy vs HF `generate(do_sample=False)` | token id 逐位相等 |
| 固定 seed | `temperature=0.7, seed=42` 跑两次；单条 vs 混 batch | 完全一致 |
| top-k=1 / top-p=1.0 | top-k=1 任意温度 vs greedy；top-p=1.0 vs 不裁剪同 seed | 结果一致 |
| 分布 | 固定 logits 采样 10k 次 | 词频 ≈ `softmax`，卡方 p>0.01 |
| penalties / logprobs | 重复 token 的 logit；`logprobs=-1` 的 `sum(exp)` | 前者被压低；后者 ≈1（atol 1e-5），sampled 的 lp == 手算 |
| detokenize | 中文 + emoji，逐 token 喂 | delta 拼接 == `decode(all_ids)` |
| stop 跨 token | `stop="\n\n"` 拆在两个 token | 命中且截断正确 |
| streaming | 收集全部 `delta_text` | 拼接 == 最终 text；`finished` 仅一次 |

**失败现象 → 原因**：和 HF 差几个 token 但都通顺 → 温度作用在惩罚之前；`top_p=1.0` 时输出
变差 → 排序后没 scatter 还原；中文流式出现 `�` → 没处理结尾 `\ufffd`；temperature=0.7 时
logprob 偏大 → 用了 processed logits；固定 seed 结果不同 → generator 每步重设或走了全局 RNG。

---

## 7. 常见坑

1. **先 temperature 再 penalties**：惩罚强度被温度缩放，`temperature>1` 等于没惩罚。
2. **top-p 排序后忘记还原**：采出来的 token id 其实是“排名”。
3. **整行 mask 成 -inf**：没有 `drop[:, 0] = False` → softmax 全 0 → 除零 NaN（“对 -inf 排序”本身不产生 NaN）。
4. **greedy 行也走 multinomial**：`temperature=0` 时先除温度会得 inf/nan。
5. **generator device 不匹配**：CUDA 张量配 CPU generator → RuntimeError。
6. **logprobs 用 processed**：`sum(exp)≠1`，与 vLLM 差一个温度因子。
7. **全量 decode 后直接当增量输出**：客户端看到重复文本；`text` 与 `delta_text` 别混用。
8. **多字节字符被切两半**：`�` 一旦吐出去就修不回来。
9. **stop string 没 buffer**：先吐了半截 stop，再截断已经晚了。
10. **EOS 出现在正文**：`skip_special_tokens=False` 会把 `<|im_end|>` 解出来。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests/test_m08_sampling.py tests/test_m08_detokenizer.py -q` 全绿；
- [ ] `pytest tests/test_m08_output.py -q -m gpu` 全绿（greedy/seed/stop/stream 四类）；
- [ ] greedy 与 HF `do_sample=False` 逐 token 一致、`temperature=0.7, seed=42` 两次一致；
- [ ] 中文 + emoji 流式无 `�`，delta 拼接等于全文；
- [ ] 能画出 §3.1 九步图并解释 penalties/temperature 顺序；
- [ ] `PROGRESS.md` 已更新，commit：`M08: sampling, logprobs, detokenizer, streaming`。

---

## 9. 交接：给 M09 的 agent

```text
nanovllm/sample/{metadata,sampler,ops}.py   新增 SamplingMetadata / Sampler / penalties·top_k·top_p·min_p
nanovllm/outputs.py                         新增 SamplerOutput；CompletionOutput 加 delta_text/stop_reason
nanovllm/engine/{detokenizer,processor}.py  新增 IncrementalDetokenizer；OutputProcessor 升级（add_request 多一个 prompt）
nanovllm/entrypoints/llm.py                 新增 LLM.stream()
```

M09 必知三件事：① `process_outputs()` 返回**累积 text**，`delta_text` 是增量（SSE 吐后者）；
② stop string 是前端判的，`reqs_to_abort` 非空时调用方必须交给 `engine_core.abort_requests()`；
③ `finish_reason` 已是字符串（`"stop"`/`"length"`/`"abort"`），OpenAI 层直接透传。

已知限制：只支持 raw logprobs；`n` 恒为 1；不做 DecodeStream（长输出 O(n²)）；随机采样只能对分布。

---

## 10. 自测题

1. 九步采样顺序？为什么 penalties 在 temperature 之前？
2. top-p 的 cumsum 截在哪项？为什么 `drop[:, 0] = False` 不能省？
3. logprobs 用 raw 还是 processed？vLLM V1 默认哪个？用错了哪个测试会暴露？
4. 增量 detokenize 怎么处理不完整 UTF-8？为什么只处理结尾的 `\ufffd`？
5. `stop_buffer_length` 为什么是 `max(len(stop)) - 1`？
6. greedy 与 random 混 batch，为什么必须 `torch.where` 而不是 if/else？

---

## 11. 教练流程（给 agent 用）

| 时间 | 做什么 | 禁止 |
|---|---|---|
| 20min | 讲 §3.1 九步 + §3.2 混合 batch，让用户复述 | 直接贴代码 |
| 30min | 读 `sampler.py` 的 `forward`/`sample`/`gather_logprobs` | 读整个文件 |
| 60min | 用户写 T1–T4，每步跑一次 `-k` | 一次写完四个 |
| 60min | 用户写 T6–T7（detokenizer 先于 OutputProcessor） | 先做 logprobs |
| 30min | T8 + 流式手工验收 | 跳过 stop 测试 |
| 40min | 对拍 + 分布测试 | 用随机采样逐 token 对拍 vLLM |
| 15min | 自测题 + `PROGRESS.md` + commit | 不跑测试就收工 |

**顺序提醒**：先 greedy + detokenizer（能吐中文），再 top-k/p，最后 logprobs（最容易对不上）。
