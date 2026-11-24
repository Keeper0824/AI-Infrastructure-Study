# M05 · Scheduler：token 预算、连续批处理、chunked prefill、抢占

> **模块目标**：写一个 FCFS、token-budget 驱动的 Scheduler——只决定“这一步每个请求算几个
> token”，不碰 GPU 和模型。
>
> **预计工时**：4–5 小时　**依赖**：M00（`Request`/`SchedulerOutput`）、M03（可先用 fake）
> **测试**：纯 CPU　**产出**：`test_m05_scheduler.py` 8 类测试全绿；`trace_m05.py` 打印
> §3.1 的 golden trace；一个 CPU 上的 `FakeModelRunner`。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests/test_m05_scheduler.py -q     # 8 passed
python scripts/trace_m05.py --scenario budget4      # 与 §6.4 逐行一致
```
以及五个能脱口而出的答案：① 为什么没有 prefill/decode 阶段；② waiting 为什么必须排在
running 之后；③ 抢占为什么把 `computed` 归零；④ 为什么先组装再 `+= n`；⑤ 中间步为什么
不能 append。

---

## 1. 为什么需要这个模块（10 分钟）

M03/M04 解决了“KV 放哪、attention 怎么读”。每步还有四个问题没人管：

```text
本步能算 N 个 token（max_num_batched_tokens），这 N 个分给谁、每人几个？分不下了，
拒绝新请求还是踢回 running？某请求这步能不能停？停了块怎么办？
```
这就是 Scheduler 的全部工作：**纯 CPU 的确定性逻辑**，输入 `Request` + `KVCacheManager`，
输出 `SchedulerOutput`。没有它只能 batch=1，长 prompt 会饿死其它请求——这正是
continuous batching 要解决的问题。接口全部冻结：

| 方向 | 结构 | 定义 |
|---|---|---|
| 进 | `Request.status` / `num_computed_tokens` / `num_tokens` | `00_architecture.md` §6.2 |
| 进 | `KVCacheManager.get_computed_blocks / allocate_slots / free / get_block_ids` | §6.4 |
| 出 | `SchedulerOutput`；回 `ModelRunnerOutput`；再出 `EngineCoreOutputs` | §6.3/§3.3/§3.4 |

**本模块不写模型、不写 attention、不写 kernel**：一个 fake runner 就够把状态机跑完。

---

## 2. vLLM 精读地图（只读这些）

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`；格式
> `vllm/v1/core/sched/scheduler.py:591 · schedule()`。行号本机校验过，漂了用 `grep -n`。
> `scheduler.py` 有 3436 行，**主路径约 200 行，其余 90% 是分支**。

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `scheduler.py:84` / `:591-602` / `:659-720` | `Scheduler` / `schedule()` / running 循环 | 四个字段；“没有 decoding phase”；`n = min(剩余, budget, max_model_len - computed - 1)` | connector 字段、async placeholder（:670-700） |
| 2 | `scheduler.py:777-857` | 分配、抢占、记账 | `while True` 重试；失败 `running.pop()`+`_preempt_request`；`victim==request` 才 break；`budget -= n` | 已调度请求的回收（:814-833） |
| 3 | `scheduler.py:900` / `:912-930` | waiting 开关 + 循环头 | **`if not preempted_reqs`**；`len(running) >= max_num_active_reqs` 就 break | `PauseState`、streaming |
| 4 | `scheduler.py:1037/1129` / `:1255-1277` | 前缀命中、`n`、分配 | `n = num_tokens - num_computed_tokens`（**不是** `num_prompt_tokens`）；`allocate_slots(..., num_hit, hit_blocks)`，`None` 就 break | 远端命中、async load |
| 5 | `scheduler.py:1345-1372` / `:1501-1522` / `:1616` | 进入 running、组装、`_update_after_schedule` | `computed=num_hit`→扣预算；**组装时 `computed` 还是旧值**，之后才 `+= n` | replay/pad_spec、connector |
| 6 | `scheduler.py:1571` / `:2026` / `:2493` | 抢占 + `update_from_output` | 抢占 free→PREEMPTED→0→prepend；取 sampled→append→`check_stop`→free→emit | stale output、logprobs/spec |
| 7 | `scheduler.py:2616/2649/2714`；`sched/utils.py:98` | `add_request`/`finish_requests`/`_free_request`；`check_stop` | 进 waiting；abort 摘队列；finish→free→`del`；eos→stop_ids→长度 | streaming、min_tokens |
| 8 | `sched/request_queue.py:75` / `output.py:232` / `kv_cache_manager.py:371` | `FCFSRequestQueue` / `SchedulerOutput` / `allocate_slots` | deque 四件套；字段注释；**只看签名与返回值** | priority queue、922 行的多 group/watermark |

建议读 `:1693`（cached 是旧值）、`:2823`（`has_requests`）、`kv_cache_manager.py:509-537`（watermark）。

**跳过**：priority、encoder、KV connector、DP/PP、spec、structured output、streaming input、
Mamba/sliding window、async scheduling、LoRA、metrics，分布在 `:84-590`、`:706-770`、
`:900-1128`、`:1145-1255`、`:1280-1340`、`:1400-1470`、`:1660-3436`。**不要通读 3436 行。**

---

## 3. 核心概念

### 3.1 没有阶段，只有“追平”

调度器眼里只有三个数（M00 §3.2）：`num_tokens`、`num_tokens_with_spec`（nano 恒等于
前者）、`num_computed_tokens`。`schedule()` 唯一的任务是给后者分配增量去追前者：

| 场景 | 本步 `n` | 是否采样 |
|---|---|---|
| 首次 prefill / 抢占后重跑 | `num_tokens - 0` | 追平才采样 |
| chunked prefill | `min(剩余, budget)` | 没追平不采样 |
| decode | `1` | 采样 |
| prefix caching | `num_tokens - num_hit` | 追平才采样 |

**三请求 golden trace**（A prompt 5 / B prompt 10 / C prompt 3，`max_tokens=4`，budget 8，
block 4，max_num_seqs 4；§6.2 的 golden trace 测试逐行对比）：

| step | scheduled | 调度后 computed A/B/C | 本步采样 | running / waiting |
|---|---|---|---|---|
| 1 | `A:5, B:3` | 5 / 3 / – | A:a1 | `[A,B] / [C]` |
| 2 | `A:1, B:7` | 6 / 10 / – | A:a2, B:b1 | `[A,B] / [C]` |
| 3 | `A:1, B:1, C:3` | 7 / 11 / 3 | A:a3, B:b2, C:c1 | `[A,B,C] / []` |
| 4 | `A:1, B:1, C:1` | 8 / 12 / 4 | A:a4（停）, B:b3, C:c2 | `[B,C] / []` |

B 在 step 1 只拿到 3 个 token（预算被 A 吃满），step 2 补 7 个才追平；C 到 step 3 才准入。
**同一个循环同时处理 prefill 和 decode**（`n=5` / `n=1`）。

### 3.2 token_budget

`token_budget = max_num_batched_tokens`；每个 running 最多吃
`num_tokens_with_spec - num_computed_tokens`，且不超过剩余预算；每调度一个请求
`num_scheduled[req] = n; token_budget -= n`。`sum <= budget` 由 `validate()` 守（M00 T6）；
vLLM 的第二个预算 `input_budget` nano 不需要。

### 3.3 running 优先：不够就抢最后一个

顺序不能反：running 的请求**已持有 KV 块**，waiting 的没有；先服务 waiting，刚到的长
prompt 就会挤掉正在 decode 的请求。不够时牺牲最后一个 running（FCFS 下最新进来的）。

```text
allocate_slots(req, n) -> None
  victim = running[-1]
  if victim is req: break                      # 没有别人可牺牲
  running.pop(); kv.free(victim); victim.reset_for_preemption()
  waiting.prepend_request(victim); preempted.add(victim.id); continue  # 重试当前 req
```

### 3.4 waiting 准入：先查前缀，失败就停

```text
if preempted_this_step: return                # scheduler.py:900
while waiting and budget > 0 and len(running) < max_num_seqs:
    hit_blocks, num_hit = kv.get_computed_blocks(req)   # 未开缓存时 ([], 0)
    n = min(req.num_tokens - num_hit, budget)           # 注意是 num_tokens
    if kv.allocate_slots(req, n, num_hit, hit_blocks) is None: break
```
为什么抢占过就整段跳过？被抢的请求刚 `prepend` 到队首，本步预算已被 running 用过一轮；
立刻再准入会“抢占—重入—再抢占”抖动。让它下一步以干净状态重新竞争。

### 3.5 chunked prefill

`n = min(remaining, token_budget)`。只有 `computed == num_tokens` 那步才“追平”、才允许
采样；前面的 chunk 是纯计算。runner 仍会为这行算 logits（指向每行最后一个 query），必须丢掉。

### 3.6 new / cached / resumed

| 分类 | 何时 | 放哪 | 发什么 |
|---|---|---|---|
| new | 首次调度（`WAITING`） | `scheduled_new_reqs` | prompt + 全量 `block_ids` + `num_computed_tokens` |
| cached | 之前调度过、本步继续 | `scheduled_cached_reqs` | 只发 `new_block_ids` + 计数 |
| resumed | 抢占后重入（`PREEMPTED`） | 同上 + `resumed_req_ids` | **全量** `block_ids`（worker 行已清） |

cached 有新块就填 `new_block_ids`，否则 `None`；resumed 必须是 `kv.get_block_ids(req_id)`。

### 3.7 `_update_after_schedule` 的时机

```text
组装 SchedulerOutput（旧 computed）──► worker 定位 input
_update_after_schedule: computed += n                # 组装之后！
```
vLLM 原注释：“...has to include the original number of scheduled tokens to determine
input IDs.”（`scheduler.py:1618-1619`）顺序反了，worker 从 `computed + n` 取 token，症状是
**输出逐渐乱码/重复**。

### 3.8 `update_from_output` 的过滤

`computed` 在 §3.7 已加过，判“追平”就是 `if req.num_computed_tokens < req.num_tokens: continue`。
vLLM 在 runner 侧用 `discard_request_mask`（`gpu_model_runner.py:2058`）清空这些 token；
nano 在 Scheduler 再判一次。**两边至少一边要做对**。

### 3.9 抢占是 recompute

`free` 块 → `computed=0` → `PREEMPTED` → `num_preemptions+=1` → `waiting.appendleft`。
不 swap KV：省拷贝、费算力；重跑可能命中 M07 的缓存，但**不能假设**。

### 3.10 停止：append → check_stop → free

```text
for tid in sampled_ids:
    req.append_output_token_ids(tid)                    # 1) 先记 token
    if check_stop(req, max_model_len): break            # 2) 再判停（设置 status）
if req.is_finished():
    kv.free(req); running.remove(req); del requests[id]; finished_req_ids.add(id)  # 3) 清理+通知
```
`check_stop` 顺序固定：eos → `stop_token_ids` → `max_model_len` / `max_tokens`。先 free
再 append 会让判停读到错的 `output_token_ids`。`ignore_eos` 不在这里判：由 LLM 初始化时
把 `sampling_params.eos_token_id` 置 `None`（与 vLLM 同义），`check_stop` 只做比较。

---

## 4. 从 vLLM 到 nanovllm

| vLLM | nanovllm | 处理 |
|---|---|---|
| `Scheduler(vllm_config, kv_cache_config, ...)` | `Scheduler(config, kv_cache_manager, max_model_len)` | 删 connector/encoder/lora/DP |
| `max_num_scheduled_tokens` + `input_budget` | 单个 `max_num_batched_tokens` | 简化 |
| `preempted_req_ids`（仅 v2 runner 用）；`update_from_output` 返回 dict | 用于 worker 清批；`-> EngineCoreOutputs` | 单 client |
| waiting 分配失败 `skip_request(); continue` | `break` | 不让一个请求被反复重排 |
| 抢占可回收“本步已调度”的请求并退预算（`:814-833`） | 只抢 `running[-1]` | 简化 |
| watermark / reserved blocks（`kv_cache_manager.py:509`）；无 `validate()` | 无 watermark；`validate()` 是 M00 新增 | 用 §7 坑 6 兜底 |

**本模块的接口**（`Scheduler` 不在 §6 公共契约里，签名由本文固定；`SchedulerOutput`
字段严格按 §6.3，一个都不加）：

```python
class Scheduler:
    def __init__(self, config, kv_cache_manager: KVCacheManager, max_model_len: int): ...
    def add_request(self, request: Request) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_output: ModelRunnerOutput) -> EngineCoreOutputs: ...
    def has_unfinished_requests(self) -> bool: ...
    def finish_requests(self, request_ids, status: RequestStatus) -> None: ...
```
内部字段：`self.waiting: FCFSRequestQueue`（deque 子类）、`self.running: list[Request]`、
`self.requests: dict[str, Request]`、`self.finished_req_ids: set[str]`（上一步完成，待放进
下一个输出）、`self.preempted_req_ids: set[str]`（本步抢占，放进本步输出）。

`has_unfinished_requests()` 必须包含第三个集合：

```python
return bool(self.waiting) or bool(self.running) or bool(self.finished_req_ids)
```
否则最后一个请求完成时 finish 通知还没送到 worker，`EngineCore` 就退出，batch 行和
`CachedRequestState` 泄漏（vLLM 用 `has_requests()` 合并这两件事）。

---

## 5. 实现任务

> 按顺序；每个任务先写失败测试再写实现。新建 `nanovllm/sched/{request_queue,utils,scheduler}.py`。

### T1 · 骨架、队列、`add_request` / `finish_requests`（30 分钟）

`FCFSRequestQueue(deque)`：`add_request`/`pop_request`/`peek_request`/`prepend_request`
分别就是 `append`/`popleft`/`self[0]`/`appendleft`。

```python
def __init__(self, config, kv, max_model_len):
    # waiting/running/requests/finished/preempted

def add_request(self, request):
    # assert id 唯一；status=WAITING；requests[id]=request；waiting.add_request
def finish_requests(self, ids, status):
    # 从 running/waiting 摘掉 → 设 status → kv.free → del requests[id] → finished.add
```
验收：`pytest tests/test_m05_scheduler.py -q -k "add_request or finish"`。
坑：① `ids is None` 要表示“全部”（M09 abort 用）；② 已完成的请求再 abort 要静默跳过；
③ 先 `assert RequestStatus.is_finished(status)`，防止传 `RUNNING`。

### T2 · running 循环（60 分钟）

```python
while i < len(self.running) and budget > 0:
    req = self.running[i]
    n = req.num_tokens_with_spec - req.num_computed_tokens
    n = min(n, budget, self.max_model_len - 1 - req.num_computed_tokens)
    if n <= 0: i += 1; continue
    blocks = self.kv.allocate_slots(req, n)
    if blocks is None:
        victim = self.running[-1]
        if victim is req: break
        self.running.pop(); self.kv.free(victim)
        victim.reset_for_preemption(); self.waiting.prepend_request(victim)
        preempted.add(victim.request_id); continue
    self._new_blocks[req.request_id] = blocks; num_scheduled[req.request_id] = n
    budget -= n; i += 1
```
验收：`pytest tests/test_m05_scheduler.py -q -k "running or budget"`。
坑：① 抢占后忘了 `continue`；② `victim is req` 时又 pop（IndexError）；③ `-1` 是给采样 token 留位置。

### T3 · waiting 循环（60 分钟）

```python
if not preempted:
    while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
        req = self.waiting.peek_request()
        hit_blocks, num_hit = self.kv.get_computed_blocks(req)
        n = min(req.num_tokens - num_hit, budget)
        if n <= 0: break
        if self.kv.allocate_slots(req, n, num_hit, hit_blocks) is None: break
        self.waiting.pop_request(); req.num_computed_tokens = num_hit
        req.status = RUNNING; self.running.append(req)
        self._new_blocks[req.request_id] = self.kv.get_block_ids(req.request_id)
        num_scheduled[req.request_id] = n; budget -= n
```
验收：`pytest tests/test_m05_scheduler.py -q -k "waiting or chunk"`。
坑：① 用了 `req.num_tokens` 而非 `num_tokens - num_hit`（M07 会重复算）；
② 准入后没设 `computed = num_hit`；③ `max_num_seqs` 用了 `<=`。

### T4 · 组装 `SchedulerOutput` + `_update_after_schedule`（40 分钟）

```python
out = SchedulerOutput(
    scheduled_new_reqs=[...],               # 本步首次调度的 WAITING
    scheduled_cached_reqs=CachedRequestData(
        req_ids=[...], resumed_req_ids={...}, new_block_ids=[...],
        num_computed_tokens=[...], num_output_tokens=[...]),   # 旧值！
    num_scheduled_tokens=num_scheduled, total_num_scheduled_tokens=sum(num_scheduled.values()),
    finished_req_ids=set(self.finished_req_ids), preempted_req_ids=set(preempted))
out.validate(self.max_num_batched_tokens); self._update_after_schedule(out); return out

def _update_after_schedule(self, out):
    for rid, n in out.num_scheduled_tokens.items():
        self.requests[rid].num_computed_tokens += n
    self.finished_req_ids.clear(); self.preempted_req_ids.clear()
```
验收：`pytest tests/test_m05_scheduler.py -q -k "output or update_after"`。
坑：① 先 `+= n` 再组装；② resumed 只发增量块；③ 忘了清 `finished_req_ids`，finish 会重复发。

### T5 · `check_stop` 与 `update_from_output`（60 分钟）

```python
def check_stop(request, max_model_len) -> bool:
    last = request.output_token_ids[-1]
    # 1) last == eos_token_id                     -> FINISHED_STOPPED
    # 2) last in stop_token_ids                   -> FINISHED_STOPPED + stop_reason
    # 3) num_tokens >= max_model_len or num_output_tokens >= max_tokens -> LENGTH_CAPPED
    # 4) 否则 False
```

```python
def update_from_output(self, out, model_output):
    outputs = EngineCoreOutputs()
    for rid, n in out.num_scheduled_tokens.items():
        req = self.requests.get(rid)
        if req is None or req.is_finished(): continue
        idx = model_output.req_id_to_index.get(rid)
        new_ids = model_output.sampled_token_ids[idx] if idx is not None else []
        if req.num_computed_tokens < req.num_tokens: continue      # §3.8
        appended = []
        for tid in new_ids:
            req.append_output_token_ids(tid); appended.append(tid)
            if check_stop(req, self.max_model_len): break
        if req.is_finished():
            self.kv.free(req); self.running.remove(req); del self.requests[rid]
            self.finished_req_ids.add(rid)  # 下一步通知 worker
        if appended or req.is_finished():
            outputs.outputs.append(EngineCoreOutput(
                rid, appended, req.get_finished_reason() if req.is_finished() else None))
    return outputs
```
验收：`pytest tests/test_m05_scheduler.py -q -k "stop or update_from_output"`。
坑：① 判“追平”写成 `>`；② 没有 `running.remove`，下一步重复调度；③ finished 留在 `requests`。

### T6 · 抢占（30 分钟）

五件事：`kv.free`、`PREEMPTED`、`computed=0`、`num_preemptions+=1`、`waiting.prepend_request`。
验收 `pytest ... -k preempt`。坑：① victim 追加到队尾；② `kv.free` 传了 id；③ 忘了 `preempted.add`。

### T7 · `FakeModelRunner`（30 分钟）

```python
class FakeModelRunner:
    def __init__(self, scheduler, next_token): ...
    def execute_model(self, out) -> ModelRunnerOutput:
        req_ids, index, sampled = [], {}, []
        for rid in out.num_scheduled_tokens:
            req = self.scheduler.requests[rid]
            req_ids.append(rid); index[rid] = len(req_ids) - 1
            caught_up = req.num_computed_tokens >= req.num_tokens   # 已加过
            sampled.append([self.next_token(req)] if caught_up else [])
        return ModelRunnerOutput(req_ids, index, sampled)
```
`next_token` 默认 `100 + req.num_output_tokens`（**确定性**，别用 `hash()`：PYTHONHASHSEED
会让 trace 漂）；eos 测试传 `lambda req: EOS`。验收：`pytest ... -k fake`。坑：① fake
runner 自己又 `+= n`；② 中间 chunk 也返回 token；③ `req_id_to_index` 与 `req_ids` 不一致。

### T8 · golden trace 与收尾（40 分钟）

`scripts/trace_m05.py` 用 fake KV 跑完整生命周期，每步打印（waiting 顺序也要打）：

```text
step=1 sched={A:5,B:3} total=8 computed={A:5,B:3} running=[A,B] waiting=[C] sampled={A:[100]}
```
验收：`trace_m05.py --scenario budget4` 与 §6.4 一致；`pytest` 8 passed。坑：① 打印调度后
的 computed；② 不打印 waiting 顺序；③ golden 用了随机 token。

---

## 6. 测试与对拍

### 6.1 fake KV：把显存变成整数

`tests/fake_kv.py` 里 30 行即可，语义必须和 M03 的 `allocate_slots` 一致（参数里的
`num_computed_tokens` 是**调度前**的值）：

```python
class FakeKVCacheManager:
    def get_computed_blocks(self, req): return ([], 0)          # M05 不开缓存
    def allocate_slots(self, req, n, num_hit=0, hit_blocks=None):
        need = ceil((req.num_computed_tokens + num_hit + n) / self.block_size)
        if need - len(self.blocks.get(req.request_id, [])) > self.free: return None
    def free(self, req): ...;  def get_block_ids(self, req_id): ...
```
M06 换真 manager 时测试不用改。

### 6.2 八类测试（参数已验证，可直接抄）

| # | 测试 | 参数 | 断言 |
|---|---|---|---|
| 1 | FCFS 顺序 | 3 个 prompt 各 3 token，`max_tokens=[2,3,4]`，budget=16 | 第 2/3/4 步依次完成 A/B/C；running 顺序不变 |
| 2 | token budget | 2 个 prompt 各 6，budget=4 | 每步 `total<=4`；A 的 prefill 是 4+2，B 是 2+3+1 |
| 3 | chunked prefill | prompt=10，budget=4 | 序列 `4,4,2`；前两步 `sampled==[]`，第三步非空 |
| 4 | 抢占 | 3 个 prompt `[6,4,4]`，blocks=4，bs=4，budget=4，max_tokens=4 | 出现 `preempted_req_ids`；victim `computed==0`、`PREEMPTED`、`num_preemptions==1`、在 `waiting[0]`；结束时 `free==4` |
| 5 | 停止条件 | eos / `stop_token_ids` / `max_tokens` / `max_model_len` 各一例 | `FINISHED_STOPPED` vs `FINISHED_LENGTH_CAPPED`；`finish_reason` 对应 |
| 6 | 不变量 | 每个场景每步 | `validate()` 不抛、`total==sum`、new/cached 不相交 |
| 7 | 无死循环 | 20 个随机请求（seed 固定），budget=8，blocks=32 | 200 步内全部完成 |
| 8 | golden trace | §3.1 三请求 | 每步 `(scheduled, computed, running, waiting, sampled)` 全等 |

**抢占用例的块数很关键**：`blocks=4` 时 A 在 step 4 挤掉 C、step 5 挤掉 B，B 从 0 重跑，
11 步结束、块全归还；`blocks=3` 会死锁（§7 坑 6）——“所有块被一个请求占满”的真实性质。

### 6.3 对拍方式

1. **golden trace**：手推表 vs `trace_m05.py` 输出，逐行 diff；
2. **不变量**：`validate()` + “每步 total <= budget” + “finished 不再被调度”；
3. **最终状态 + 和 M06 对拍**：`output_token_ids` 与“逐个跑”一致；`req_id_to_index` 能解释
   `req_ids`（先用 fake 固定契约）。

### 6.4 golden 表（`--scenario budget4`，A/B prompt 各 6、max_tokens=8）

```text
step 1  sched={A:4}        total=4  computed={A:4}        waiting=[B]
step 2  sched={A:2,B:2}    total=4  computed={A:6,B:2}    waiting=[]
step 3  sched={A:1,B:3}    total=4  computed={A:7,B:5}    waiting=[]
step 4  sched={A:1,B:1}    total=2  computed={A:8,B:6}    waiting=[]
```
失败现象 → 先查：

| 现象 | 先查 |
|---|---|
| 第 2 步就吐 2 个 token / computed 涨得比 scheduled 多 | §3.8 过滤没写；fake KV 或 scheduler 重复 `+= n` |
| 抢占后 victim 还在 running / 永不结束 / `validate()` 挂 | `running.pop()` 位置；`check_stop`；`num_scheduled` 记成“剩余” |

---

## 7. 常见坑

1. **组装前就 `+= n`**：worker 从 `computed+n` 取 input，第一步就错位。修复：把
   `_update_after_schedule` 放在组装之后、返回之前。
2. **抢占后没重试当前请求 / 把当前请求也抢占**：正确是 `continue` 且不动 `i`；
   `victim is req` 时必须 `break`，否则 `running` 清空、无限抢自己。
3. **waiting 用了 `request.num_tokens`**：M05 看不出；M07 一开缓存，命中 8 个 token 的请求
   会被要求再算 8 个。
4. **中间 chunk append 采样 token**：输出比预期长且错位。Scheduler 判
   `computed >= num_tokens`，runner 判 `computed + n >= num_tokens`，两边都写才稳。
5. **`finished_req_ids` 没放进下个输出 / 完成请求没从 `running`、`requests` 移除 / 预算扣
   两次**：batch 行只增不减（OOM）、“已完成还在生成”、以及 `validate()` 抓不到的“少调度”。
6. **全部块被占满导致空转**：唯一 running 请求还需新块、而它就是 `running[-1]`，`break`
   后 waiting 也分配不到，本步 `total==0` 且无 finish → 死循环。nano 在 `EngineCore.step()`
   自检 `if total==0 and not finished_req_ids: raise RuntimeError`；vLLM 用 watermark 降低
   概率（`kv_cache_manager.py:509-537`）但没有消除。
7. **`max_num_seqs` 只查 waiting**：准入后立刻查，否则 batch 行数超过 `InputBatch` 容量。

---

## 8. 完成定义（DoD）

- [ ] `python -m pytest tests/test_m05_scheduler.py -q` → 8 passed；
- [ ] `python scripts/trace_m05.py --scenario budget4` 与 §6.4 逐行一致；
- [ ] running 循环在 waiting 之前，且有 `if not preempted` 守卫；
- [ ] `_update_after_schedule` 在组装之后，并在其中清空 `finished_req_ids`（§3.7）；
- [ ] `update_from_output` 先 append 再判停，再 free/移除/记录；中间 chunk 的 token 被丢弃；
- [ ] `has_unfinished_requests()` 含 finished；每个 `SchedulerOutput` 都过 `validate(budget)`；
- [ ] `PROGRESS.md` 已更新；commit：`M05: token-budget scheduler (chunked prefill / preemption)`。

---

## 9. 交接：给 M06

**新增文件**：`nanovllm/sched/{request_queue,utils,scheduler}.py`、`tests/fake_{kv,runner}.py`、`scripts/trace_m05.py`；**公开 API** 是 §4 的六个方法，三个数据结构未改。

**给 M06 的四条契约**（写代码前先对齐）：

1. `num_scheduled_tokens` 的 key 集合 = worker 本步必须有 batch 行的请求集合；
2. `NewRequestData`/`CachedRequestData` 的 `num_computed_tokens` 都是**本步起点**（旧值）；
3. runner 对“没追平”的行必须回 `[]`，且**不要**把采样 token 追加进 worker 状态；
4. `finished_req_ids` 会在某个后续 `SchedulerOutput` 里出现一次，worker 要能处理“本步 0 token”。

**已知限制**：无 priority/spec/encoder/connector/watermark；抢占只抢 `running[-1]`；
`enable_chunked_prefill=False` 时 waiting 只收“一个 chunk 装得下”的请求。

---

## 10. 自测题（做完后不看代码回答）

1. 为什么没有 prefill/decode 阶段？`num_computed_tokens` 追的是哪个量？§3.1 里 B 为什么是 3+7？
2. running 和 waiting 的顺序为什么不能反？
3. 抢占时为什么把 `computed` 归零？为什么不把 KV swap 到 CPU？
4. 为什么 `_update_after_schedule` 必须在组装 `SchedulerOutput` 之后？
5. chunked prefill 中间步 runner 有 logits，为什么不能 append？
6. `finished_req_ids` 的作用？为什么 `has_unfinished_requests()` 要把它算进去？

---

## 11. 教练流程（给 agent 用）

| 时间 | 做什么 | 禁止 |
|---|---|---|
| 25min | 讲 §3.1/§3.2；用户先手推 §3.1 的表再对照 | 先讲抢占 |
| 30min | 读 `scheduler.py:591-720`（running 循环） | 读 `:84-590` |
| 30min | 读 `:900-940` + `:1129` + `:1255-1272`（waiting 循环） | 读 connector 分支 |
| 90min | 写 T2/T3，再写 T4/T5，用 `validate()` 当第一道防线 | 一次写 T2+T3+T4 |
| 60min | T7 fake runner + 8 类测试 + golden trace | 上 GPU |
| 15min | 复盘 §7 坑 1/4/6 + 更新 `PROGRESS.md` | 直接贴参考实现 |

顺序原则：**先画 trace，再写代码**。用户画不出 §3.1 的表，就不要开始 T2。
