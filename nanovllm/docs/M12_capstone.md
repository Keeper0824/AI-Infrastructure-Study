# M12 · Capstone：对拍、性能报告、代码地图

> **模块目标**：把 M00–M11 收束成一份可展示的成果：正确性矩阵、性能矩阵、代码地图、
> `REPORT.md`，以及 `v1.0` tag。
> **预计工时**：2–3 小时　**依赖**：M00–M10（M11 可选）　**测试**：全量 + GPU
>
> **M11 可以跳过**：本模块不依赖多进程；矩阵里不包含 IPC 行，报告写“单进程版本”即可。
> **本模块不加新功能**：只验证、测量、写作。任何“顺手优化”都放到 tag 之后。

---

## 0. 本模块结束时你会有什么

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests -q
python -m pytest tests -q -m gpu
python benchmarks/bench_m12_correctness.py --model models/Qwen3-0.6B
python benchmarks/bench_m12_matrix.py --model models/Qwen3-0.6B --suite core
python benchmarks/bench_m12_profile.py --model models/Qwen3-0.6B --batch 1
```

以及：`docs/CODE_MAP.md`、`REPORT.md`、`git tag v1.0`、更新后的 `README.md` / `PROGRESS.md`。
五个能脱口而出的答案：TTFT/TPOT 各反映什么、为什么要 warmup + 同步、瓶颈在哪、先改哪三处、
哪三个 vLLM 机制删得最狠。

---

## 1. 为什么需要这个模块（10 分钟）

前 11 个模块每完成一个，你只验证了一个局部。Capstone 要回答三个全局问题：

1. **它真的对吗？** 不是“跑通了一个 prompt”，而是模型/注意力/调度/采样每一层都有 oracle。
2. **它慢在哪？** 不是“比 vLLM 慢”，而是慢在 GPU kernel、Python 调度、采样还是 detokenize。
3. **它和 vLLM 差在哪？** 不是文件数，而是每个机制保留了什么、删了什么、为什么能删。

三个交付物对应这三个问题：**正确性矩阵**（对不对）、**性能矩阵 + profile**（慢在哪）、
**代码地图 + REPORT**（差在哪、学到什么）。M12 的“测试”不是再写功能测试，而是让这些
结论**可复现、有数据、能讲给别人听**。

---

## 2. 精读地图：回读六个锚点

> 参照树：`/home/dictator/ai_infra/target/vllm-0.32.0/`。只读下表区间，不重读整文件。

### 2.1 必读（每个 10–20 分钟）

| # | 位置 | 符号 | 看什么 | 可跳过 |
|---|---|---|---|---|
| 1 | `vllm/v1/engine/core.py:657` | `EngineCore.step()` | 五步：`schedule` → `execute_model(non_block=True)` → 必要时 `sample_tokens` → `_process_aborts_queue` → `update_from_output` | grammar、stats、DP、`post_step` |
| 2 | `vllm/v1/core/sched/scheduler.py:591` | `Scheduler.schedule()` | 顶部“没有 prefill/decode 阶段，只有追平”的注释；RUNNING 循环（`:659`）、WAITING 循环（`:899`）、组装 `SchedulerOutput`（`:1501`） | encoder/connector/spec/priority/streaming |
| 3 | `vllm/v1/core/kv_cache_manager.py:371` | `allocate_slots()` | 签名 + `Blocks layout` 注释（`:420`）；对照 M03 的 10 步算法 | 多 group、sliding、connector、external tokens |
| 4 | `vllm/v1/worker/gpu_model_runner.py:1920` | `_prepare_inputs()` | `req_indices`（`:1944`）、`positions`（`:1952`）、`token_indices`（`:1967`）、`logits_indices = query_start_loc[1:] - 1`（`:2191`） | M-RoPE、spec、PP、预分配 buffer 优化 |
| 5 | `vllm/v1/attention/ops/triton_unified_attention.py:179` | `kernel_unified_attention` | 参数里的 block table / seq lens / strides；2D 与 3D 两条路径的差别 | FP8、sinks、alibi、sliding、多模态 |
| 6 | `vllm/v1/engine/llm_engine.py:319` | `LLMEngine.step()` | `get_output` → `process_outputs` → abort stop strings → stats；前端只做输出处理 | renderer、mm cache、metrics |

**一句话对照**：vLLM 的 `EngineCore.step()` 是“五步 + 异步 future”；nano 是同样的五步，但
`execute_model()` 同步返回、`sample_tokens()` 紧随其后。你不需要在 M12 改回异步，只要在
REPORT 里说清这个差异对测量意味着什么（CPU/GPU 不重叠，TTFT 偏高）。

### 2.2 建议读（选读）

| 位置 | 符号 | 为什么 |
|---|---|---|
| `vllm/v1/engine/output_processor.py:681` | `process_outputs()` | 对照 nano 的前端输出处理；只看入参/出参 |
| `vllm/v1/sample/sampler.py:22` | `Sampler` | 对照 M08 的采样顺序；不要重读全部 |
| `vllm/v1/worker/gpu_worker.py:634` | `determine_available_memory()` | 报告显存预算时解释 `num_gpu_blocks` 的来源 |
| `vllm/v1/worker/gpu_model_runner.py:6482` | `capture_model()` | 只在报告 graph 一节时看 bucket 思路 |

### 2.3 明确跳过

Scheduler 的 3400 行、`gpu_model_runner.py` 的 7200 行、`EngineCoreProc` 生命周期、DP/EP、
KV connector、多模态、spec decode、量化、CUDA graph 的完整 dispatcher。**M12 不读新机制**，
只回读你已经实现过的六个点。

---

## 3. 核心概念

### 3.1 正确性矩阵：每一格都有 oracle 和 assert

| 格子 | oracle | 比较对象 | 断言 |
|---|---|---|---|
| A 模型前向 | HF `Qwen3ForCausalLM`（eager, bf16） | 固定 prompt 的 last-token logits | `assert_close`，阈值沿用 M01 |
| B 连续 cache | M02 的 `past_key_values` 生成 | M03/M04 的 paged 生成 | greedy token 完全相同 |
| C naive paged | B 的 logits | M04 naive backend | `assert_close`，阈值沿用 M04 |
| D Triton paged | C 的 logits | M04 Triton backend | `assert_close`；边界长度也要测 |
| E 批量 vs 逐个 | 逐个 `generate()` | batch=2/8 `generate()` | greedy token 逐 token 相同 |
| F prefix on/off | cache off 的 greedy 输出 | cache on 的 greedy 输出 | 完全相同；且命中率 > 0 |
| G chunked on/off | 一次性 prefill | `max_num_batched_tokens` 很小时 | 输出相同 |
| H graph on/off | eager decode | M10 graph decode | 输出相同（若做了 M10） |

规则：**阈值是脚本里的常量**（`ATOL`/`RTOL`），不是每个 cell 临时拍脑袋；每个 cell 输出
`name / passed / detail / metrics`，最后统一 `assert`。能复用就复用：E 格来自 M06 的
batch-vs-sequential 测试，C/D 格来自 M04 的对拍，F 格来自 M07；矩阵只负责汇总，不重写 oracle。

### 3.2 性能指标：定义先写死

```text
TTFT_i  = t_first_token_i - t_submit_i
TPOT_i  = (t_last_token_i - t_first_token_i) / (n_output_i - 1)
throughput = sum(n_output_i) / (t_end - t_start)
```

- `t_submit`：`add_request` 之前还是之后？**全课程统一为“进入 `LLM.generate/stream` 之前”**，
  并在报告里写明是否含 tokenization；
- `t_first_token`：`stream()` 第一次 yield 该请求输出的时刻；不是引擎内部 sample 的时刻；
- `n_output_i` 含首 token；`n_output_i == 1` 时 TPOT 记为 `None`，不要除零；
- 用 `time.perf_counter()`，不要 `time.time()`；计时窗口前后 `torch.cuda.synchronize()`；
- 报告 `min / median / p95`；3 次重复时 p95 等于最大值，要在脚注说明。

### 3.3 公平 benchmark 的检查清单

- 同一 GPU、同一 dtype（bf16）、同一模型权重、同一 tokenizer；
- 固定 prompt 集合（固定 seed 生成或固定文本文件），固定 `max_tokens`，greedy；
- 每个配置至少 1 次 warmup（不计入），再跑 3 次；首次包含 Triton autotune / graph capture；
- batch 对比时，**每个请求的 output 长度相同**；不同长度要单独列，不要混进同一个吞吐数；
- 关掉其他占 GPU 的进程；记录 GPU 型号、驱动、torch/Triton 版本；
- 如果要和 vLLM 比：必须同模型/同 batch/同采样参数；没有同一套环境就只做机制解释、不贴数字。
  vLLM 默认开 CUDA graph/FlashAttention，nano 不追求赢，只解释差距来源。

### 3.4 Amdahl 与 profile：先找瓶颈，再谈优化

一步的时间可以粗分为：

```text
step = CPU(schedule + sample + detokenize + Python) + GPU(forward) + IPC(若 M11)
```

如果 CPU 部分和 GPU 部分相当，甚至更大，那么换更快的 attention kernel 收益有限——这就是
Amdahl 定律：加速占比 20% 的部分 2 倍，总时间只降到 `1/(0.8 + 0.2/2) ≈ 1.11` 倍。

用 `torch.profiler` 看 top CUDA op；用 `perf_counter` 分别包住 `schedule()`、
`execute_model()+sample_tokens()`、`process_outputs()`；用 CUDA event 看 GPU 总时间。
**先证明瓶颈在哪，再写“后续优化”**；没有 profile 的优化建议一律不写进报告。

### 3.5 代码地图：机制级，不是文件列表

代码地图的每一行回答：“这个 vLLM 机制在 nano 的哪一行？删了什么？为什么能删？”
不是“nano 有哪些文件”。至少覆盖：连续批处理、paged KV、prefix caching、chunked prefill、
抢占、采样、增量 detokenize、CUDA graph、多进程；以及明确删除的 TP/PP/DP、spec、量化、
多模态、LoRA、结构化输出、KV connector、ZMQ/共享内存。

### 3.6 报告叙事：claim → evidence → why

每个结论写成三句话：**声称什么**（数字）、**证据在哪**（矩阵 cell / 结果文件 / 命令）、
**为什么**（机制解释）。反例：“nano 比 vLLM 慢 3 倍”不是结论；“batch=8、output=128 时
nano 吞吐是 vLLM 的 1/3，profile 显示 42% 时间在 Python 调度与逐 token detokenize”才是。

---

## 4. 从 vLLM 到 nanovllm：本模块只读不写

| vLLM 机制 | nano 的位置 | 处理 |
|---|---|---|
| `EngineCore.step()` 五步 | `engine/core.py` | 保留；同步版，去掉 future/grammar/stats |
| `Scheduler.schedule()` token 预算 | `sched/scheduler.py` | 保留主路径；删 encoder/connector/spec/DP/priority |
| `allocate_slots()` 分页分配 | `core/kv_cache_manager.py` | 保留 10 步；删多 group/sliding/connector |
| `_prepare_inputs()` 组 batch | `worker/model_runner.py` | 保留 10 行流水线；删 M-RoPE/spec/PP/预分配优化 |
| `kernel_unified_attention` | `attention/ops/paged_attention.py` | 保留一个简化 kernel；删 2D/3D、FP8、sinks |
| `LLMEngine.step()` 前端 | `engine/llm_engine.py` | 保留 get_output → process_outputs |
| `AsyncLLM` / OpenAI | `engine/async_llm.py`、`entrypoints/` | 保留最小流式；删 metrics/鉴权/多模态 |
| TP/PP/DP、spec、量化、LoRA、多模态、KV connector | 无 | 删除；在代码地图里逐项写原因 |

**本模块不新增公共接口**：只新增 `benchmarks/*.py` 脚本与 `docs/CODE_MAP.md`、`REPORT.md`。
`LLM.generate/stream`、`EngineCore.step`、`SchedulerOutput` 等冻结接口一律不动。

---

## 5. 实现任务

### T1 · 正确性矩阵脚本（60 分钟）

文件：`benchmarks/bench_m12_correctness.py`；结果：`benchmarks/results/correctness.json`。

```python
@dataclass
class Cell:
    name: str
    passed: bool
    detail: str
    metrics: dict

def cell_hf_logits(model_path: str) -> Cell: ...
def cell_backends(model_path: str) -> Cell: ...
def cell_batch_vs_sequential(model_path: str) -> Cell: ...
def cell_prefix_cache(model_path: str) -> Cell: ...
def cell_chunked_prefill(model_path: str) -> Cell: ...

def main() -> int:
    cells = [...]
    for c in cells:
        assert c.passed, f"{c.name}: {c.detail}"
    write_json(...); print_table(cells)
    return 0
```

验收：

```bash
python benchmarks/bench_m12_correctness.py --model models/Qwen3-0.6B \
    --out benchmarks/results/correctness.json
# exit 0；表里 A–G 全 pass
```

做一次“测试的测试”：临时注释掉 `KVCacheManager.cache_blocks()` 的调用（或把 prefix cache
关闭），确认 F 格变红；恢复后再跑。矩阵必须能抓住真实回归。

坑：① 比较了不同位置的 logits（HF 取最后一个 prompt token，nano 也必须是同一个）；
② 每个 cell 复用同一个引擎，prefix cache 的残留状态污染下一格；③ bf16 用默认
`assert_close` 阈值，误报/漏报。

### T2 · 性能矩阵脚本（60 分钟）

文件：`benchmarks/bench_m12_matrix.py`；结果：`benchmarks/results/perf.json` + `perf.md`。

CLI 至少支持：

```bash
python benchmarks/bench_m12_matrix.py --model models/Qwen3-0.6B --suite core \
    --batch-sizes 1 8 --prompt-lens 128 512 --output-lens 32 128 \
    --repeats 3 --warmup 1 --out benchmarks/results/perf.json
```

**不要重写计时循环**：M04 已有 `bench_m04_attention.py`，M07 有 `bench_prefix_cache.py`，
M10 有 `bench_m10_decode.py`（TTFT/TPOT/throughput、graph on/off），M06 交接的 engine bench
覆盖 batch=1/8。矩阵脚本只做
三件事：拼配置网格、调用这些脚本（import `run_case` 或 subprocess 解析输出）、汇总 JSON/MD。
旧脚本没有机器可读输出时，给它们加 `--json PATH`（只加输出，不改计时逻辑），不要复制
warmup/sync 代码。矩阵只补旧脚本没覆盖的轴：batch vs sequential、prefix on/off、混合长度。

`core` 套件：graph off、prefix off，batch × prompt × output；`graph` 套件直接引用 M10 的结果；
`prefix` 套件测 cache on/off 的 TTFT。每个配置：固定 seed 构造 prompt 集（不同前缀）；warmup
1 次；重复 3 次记录每个请求的 TTFT/TPOT/output 长度；`synchronize()` 后取 `t_end`；聚合
`min/median/p95`，JSON 里写 `git rev`、GPU、torch/Triton 版本。

验收：`perf.json` 每个 case 有 3 个 repeat、`n_output` 与请求一致、`throughput > 0`；
`perf.md` 的数字与 JSON 对得上。

坑：① 第一次运行包含 Triton autotune/graph capture；② 不同 batch 用了不同 output 长度；
③ 用 `generate()` 而不是 `stream()`，根本拿不到 TTFT。

### T3 · profile：找出瓶颈（40 分钟）

文件：`benchmarks/bench_m12_profile.py`；结果：`benchmarks/results/profile.txt`。

```bash
python benchmarks/bench_m12_profile.py --model models/Qwen3-0.6B --batch 1 --steps 20
```

输出必须包含：

- `torch.profiler` 的 CUDA top-15 op 表（按 `cuda_time_total`）；
- 每步的 `cpu_step_ms`、`gpu_kernel_ms`、`cpu_share`；
- 分项：`schedule_ms`、`execute_sample_ms`、`process_outputs_ms`；
- 一句自动判定：`cpu_bound` / `gpu_bound`（例如 CPU 占比 > 60% 记为 CPU-bound）。

验收：表非空；分项之和与 `cpu_step_ms` 同一量级（允许 profiler 开销，报告里注明）；
结论能解释 T2 中的一个数字。

坑：① profiler 本身改变时序；② 把 warmup 步算进去；③ 开着 CUDA graph 做 CPU profile，
看到的不是真实 launch 开销。

### T4 · 代码地图（30 分钟）

文件：`docs/CODE_MAP.md`。模板：

```markdown
| nano 文件 | vLLM 参照（符号优先） | 保留 | 删除 / 简化 | 证据（测试/行号） |
|---|---|---|---|---|
| request.py | vllm/v1/request.py:60 · Request | 三个计数、status | 多模态/LoRA/structured | test_m00_request |
| sched/scheduler.py | scheduler.py:591 · schedule | token 预算/抢占 | encoder/connector/spec/DP | test_m05_* |
| core/kv_cache_manager.py | kv_cache_manager.py:371 · allocate_slots | 10 步分配 | 多 group/sliding | test_m03_* |
| attention/ops/paged_attention.py | triton_unified_attention.py:179 | 单 kernel | 2D/3D/FP8/sinks | test_m04_* |
| engine/core.py | engine/core.py:657 · EngineCore.step | 五步 | future/grammar/stats | test_m06_* |
```

再加一张“机制 → 删/留”表（TP/PP/DP、spec、量化、多模态、LoRA、结构化输出、KV connector、
ZMQ/共享内存），每行写“为什么可以删”。

验收：

```bash
for f in $(find nanovllm -name '*.py' | sort); do
  grep -q "$f" docs/CODE_MAP.md || echo "MISSING $f"
done
# 无输出
```

坑：① 只列文件、不写删了什么；② 引用未 grep 验证的行号；③ 把“没实现”写成“简化”，
读者以为支持。

### T5 · `REPORT.md`（60 分钟）

文件：`REPORT.md`。固定骨架：

```markdown
# nanovllm REPORT
## 0. TL;DR（3 条，每条带数字）
## 1. 环境与复现（GPU/torch/Triton/模型/命令）
## 2. 正确性矩阵（A–H 表 + 阈值）
## 3. 性能矩阵（batch/prompt/output、graph、prefix）
## 4. 瓶颈分析（profile top ops + CPU/GPU 占比）
## 5. 与 vLLM 的差距（可选；机制级解释）
## 6. 学到的 5 件事（每条对应一个模块/一次踩坑）
## 7. 后续可做的 3 件事（TP / spec / quant，各一句）
## 8. 已知限制（明确列出没做的）
```

每个数字后面标注来源（`correctness.json#cell-F`、`perf.json#case-...`、`profile.txt`）。
验收：随机抽 5 个数字都能在结果文件里找到；每个“快/慢”结论都有机制解释。

坑：① 只写“快了多少”，不写为什么；② 把 vLLM 的 TP=1/batch=1 与 nano 的 batch=8 直接比；
③ 报告数字与最终代码不是同一个 commit。

### T6 · README / PROGRESS 收尾（20 分钟）

- `README.md`：在“模块地图”后加“结果”一节，链接 `REPORT.md` 与 `docs/CODE_MAP.md`；
  标注 M11 可选；把“初始 failed”改成“M00 初始为 failed，完成后全绿”。
- `PROGRESS.md`：写“课程完成”，列出 13 个模块状态、已知限制、下一步建议（TP/spec/quant）。

验收：

```bash
grep -n "REPORT.md\|CODE_MAP.md" README.md
tail -30 PROGRESS.md
```

坑：① 改冻结接口文档；② README 链接写相对路径但文件不存在；③ PROGRESS 只写“完成”，
没写下一步入口。

### T7 · tag v1.0（10 分钟）

```bash
git status --short          # 必须为空
git add -A && git commit -m "M12: capstone (matrix / report / code map)"
git tag -a v1.0 -m "nanovllm v1.0: Qwen3-0.6B + paged attention + continuous batching"
git describe --tags --exact-match   # v1.0
```

验收：`git status` 干净；`git tag -l` 含 `v1.0`；`git show v1.0 --stat` 包含 REPORT/CODE_MAP。

坑：① 先打 tag 再提交报告；② 用 lightweight tag，别人无法看到说明；③ tag 打在临时分支上。

---

## 6. 测试与对拍

| 检查 | 命令 | 通过标准 |
|---|---|---|
| 全量单测 | `pytest tests -q` | 全绿 |
| GPU 测试 | `pytest tests -q -m gpu` | 全绿 |
| 正确性矩阵 | `python benchmarks/bench_m12_correctness.py ...` | 每个 cell `assert` 通过 |
| 矩阵有效性 | 临时破坏 prefix cache 后重跑 | F 格必须变红 |
| 性能矩阵 | `python benchmarks/bench_m12_matrix.py ...` | 每 case 3 次重复 + 中位数 |
| profile | `python benchmarks/bench_m12_profile.py ...` | top op 表 + CPU/GPU 占比 |
| 代码地图 | §5 T4 的 `for f in ...` 命令 | 无 `MISSING` |
| 报告 | 人工抽查 5 个数字 | 都能在结果文件/命令中找到 |
| 收尾 | `git status --short`、`git describe` | 干净、tag 正确 |

常见失败：矩阵某格“差一点”（阈值/位置错）；benchmark 方差大（没 warmup/没同步/有其他进程）；
报告数字对不上（用了旧结果）；代码地图漏文件（grep 循环能抓到）。

---

## 7. 常见坑

1. **没 warmup**：第一轮包含权重加载、Triton autotune、graph capture，数字全偏大。
2. **没同步 GPU**：`perf_counter` 在 kernel 入队后就返回，计时偏小。
3. **不同 batch 用不同 output 长度**：吞吐不可比；固定 `max_tokens`，或按 token 归一化。
4. **vLLM 对比不公平**：版本/torch/CUDA/模型/batch/采样参数必须一致；TP=1 vs batch=1 不是
   同一件事。没有独立环境就不要放 vLLM 数字，只做机制解释。
5. **报告只有“快了多少”**：没有 profile、没有机制解释，等于没写。
6. **代码地图漏简化项**：读者会以为 nano 支持 TP/量化/多模态。
7. **prefix cache 热状态污染**：on/off 两组要用新引擎或显式清空块池。
8. **矩阵阈值临时改**：为了变绿放宽阈值，等于自欺；阈值必须来自 M01/M04 并写进报告。
9. **tag 早于报告**：tag 指向的 commit 里没有最终数字。

---

## 8. 完成定义（DoD）

- [ ] `pytest tests -q` 与 `pytest tests -q -m gpu` 全绿；
- [ ] 正确性矩阵 A–G（做了 M10 再加 H）每格都有 `assert`，且破坏性实验能让对应格变红；
- [ ] 性能矩阵每个数字有 3 次重复 + 中位数，JSON/Markdown 一致；
- [ ] `profile.txt` 有 top op 和 CPU/GPU 占比，能解释一个性能数字；
- [ ] `docs/CODE_MAP.md` 覆盖所有 `nanovllm/*.py`，且机制级“删/留”表完整；
- [ ] `REPORT.md` 含结果、瓶颈、5 件事、3 个后续方向，每个结论有数据来源；
- [ ] `README.md` / `PROGRESS.md` 已更新；`git status` 干净；`v1.0` tag 已打；
- [ ] 没有新增公共接口、没有在 M12 改功能代码。

---

## 9. 交接：课程完成

- **下一步不是新模块**：在 `PROGRESS.md` 写“课程完成”，并给出三个方向：
  **TP/PP**（多卡切分，从 `MultiprocExecutor` 回读）、**spec decode**（`num_tokens_with_spec`
  已经留好字段）、**quant**（先做 weight-only，再对拍 logits）。
- **保留的 oracle**：M02 连续 cache、M04 naive backend、M00 契约测试。后续改任何优化，
  先跑这三层。
- **不要做的事**：为了“更像 vLLM”引入 ZMQ/msgspec/完整 EngineCoreProc；nano 的价值是
  骨架清楚，不是功能多。
- **给下一位读者**：先读 `REPORT.md` §0/§4/§8，再读 `docs/CODE_MAP.md`，最后按需回模块文档。

---

## 10. 自测题

1. TTFT 和 TPOT 分别反映什么瓶颈？
2. 为什么 benchmark 必须 warmup + 同步？
3. 你的 nano 最大的性能瓶颈在哪？怎么证明？
4. 如果要把 nano 的性能提高 2 倍，你先改哪三处？
5. 代码地图里，哪三个 vLLM 机制是你删得最狠的？为什么可以删？

---

## 11. 教练流程（给 agent 用）

先跑全量测试 → 写正确性矩阵 60min → 写 benchmark 60min → profile 40min →
写报告 60min → 代码地图 30min → 收尾 20min。

要求：先让用户跑一遍现有测试并记录基线；矩阵先做 A–D（模型/注意力），再做 E–G（调度/
缓存）；benchmark 先 batch=1，再扩到 8；profile 先 batch=1、output=128，确保能看出 CPU/GPU
占比；报告最后写，避免“先有结论再找数据”。

**禁止**：在 M12 加新功能；只测一个 prompt 就写报告；用旧结果填新报告；为了好看改阈值；
建议安装 vLLM 本体。M11 跳过的用户，报告里明确写“单进程版本”，不要伪造 IPC 数据。
