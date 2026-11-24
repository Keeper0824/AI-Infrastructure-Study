# nanovllm · 从零实现 vLLM V1

> 用大约 vLLM 1/10 的代码，按 **vLLM v0.32.0 的 V1 架构**，实现一个能跑
> Qwen3-0.6B 的迷你推理引擎。
>
> 原则：**等结构、缩规模**。类名、文件位置、数据流尽量和 `vllm/v1/` 1:1；
> 删掉的是分支和优化，不是骨架。

---

## 这是什么

一套**分模块的源码阅读 + 动手实现课程**。每个模块：

- 只让你读 vLLM 里与该功能相关的 **100–400 行**（不是整个文件）；
- 给出要实现的类/函数/数据结构的**冻结接口**；
- 给出可运行的**验收测试**和“对拍”方法；
- 以一个新 agent 窗口为单位，文档本身就是那个 agent 的教案。

最终你会得到一个：

- 支持 Qwen3-0.6B 的纯 PyTorch 模型实现；
- PagedAttention（naive + Triton 两个后端）；
- token 预算调度器（连续批处理 / chunked prefill / 抢占）；
- BlockPool + KVCacheManager + 前缀缓存；
- temperature/top-k/top-p/penalty/logprobs + 增量 detokenize；
- `LLM.generate()` 离线 API + OpenAI 兼容 HTTP/SSE 服务；
- 可选的 CUDA graph 和多进程 EngineCore。

## 这不是什么

- 不是 vLLM 的 fork，不追求性能超过 vLLM；
- 不覆盖多模态、LoRA、量化、MLA/Mamba、TP/PP/DP、KV connector、结构化输出；
- 不需要装 vLLM 本体（它要求 torch 2.13，本机是 2.11；我们只读 `/home/dictator/ai_infra/target/vllm-0.32.0/`）。

---

## 快速开始

### 0. 环境（重要：统一用 conda 环境 `vllm_learn`）

本课程统一在 **`vllm_learn`** 里跑。它已经有：

```text
Python 3.11.15
torch 2.11.0+cu130
GPU: RTX 5080 16GB
Triton 3.6.0
transformers 5.10.1
safetensors 0.8.0
```

但 **`vllm_learn` 里默认没有 pytest**（base 环境有 pytest，但 base 的 Python 3.13 没有这套
torch/transformers，不要用 base 跑 M01+）。先在你的终端里装：

```bash
conda activate vllm_learn
python -m pip install pytest
# 下载模型用；已有 HF 镜像/缓存可跳过
python -m pip install modelscope
```

验证环境：

```bash
cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py
```

`doctor.py` 会告诉你当前用的是哪个 Python、pytest 有没有、tests 能不能 collect。它报错时
不要继续，先把报错贴给 agent。

> 不要 `pip install vllm`。vLLM 0.32.0 依赖 torch 2.13，会把你的 torch 升级掉。
> 我们只读它的源码。

### 1. 下载 Qwen3-0.6B

优先用 ModelScope（国内网络更稳）：

```bash
cd /home/dictator/ai_infra/nanovllm
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
```

如果 ModelScope 不可用，再用 HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

> 模型约 1.5GB。下载后确认 `models/Qwen3-0.6B/config.json` 和
> `model.safetensors`（或分片 + index）存在。

### 2. 跑 M00 的测试

```bash
conda activate vllm_learn
cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py
cd /home/dictator/ai_infra/nanovllm && python -m pytest tests -q
```

初始状态是 **31 failed, 3 passed**，这是**故意的**：`nanovllm/` 下有 `TODO(M00-x)` 的桩函数。
你的任务是把它们实现到 34 passed。参考解在 `solutions/m00/`，**卡住 30 分钟以上再看**。

如果报 `No module named pytest`，回到 §0 执行 `python -m pip install pytest`；
如果报 `tests: No such file or directory`，先 `pwd; ls`，你大概率不在 `nanovllm/` 根目录，
或者把 `cd` 和 `pytest` 粘成了一行（要用 `&&`）。

### 3. 开始一个模块

```bash
cat docs/M00_contracts.md
```

然后按下一节的模板开一个新的 agent 窗口。

---

## 如何开一个新的 agent 窗口

每个模块都新开一个窗口，把下面这段贴进去，只改 `<模块号>`：

```text
你是「nanovllm 课程」的模块教练。工作区：/home/dictator/ai_infra/nanovllm
环境：conda 的 vllm_learn，不是 base。所有命令都先 `conda activate vllm_learn`。

请先读：
1. docs/00_architecture.md
2. docs/<模块号>_*.md
3. PROGRESS.md
4. 仓库里已有的代码和测试

然后严格按模块文档里的「教练流程」指导我，规则：
- 先跑 `cd /home/dictator/ai_infra/nanovllm && python scripts/doctor.py`，确认 Python/pytest/tests 都对；
- 一次只讲一个概念，我复述清楚再往下；
- 不要直接给我完整实现；先给接口、伪代码、失败测试，让我自己写；
- 每个任务都要有验收命令，测试不过不算完成；
- 引用 vLLM 代码必须带 文件:行号，不确定就用 grep 重定位；
- 只读模块文档指定的区间，不要让我通读整个文件；
- 结束前帮我更新 PROGRESS.md，并给一个 commit message。

如果 doctor.py 报 `No module named pytest`，先停下来让我执行：
`conda activate vllm_learn && python -m pip install pytest`，
装好再继续。

现在从模块文档的 §0 开始；先按上面的命令跑 doctor.py，
再带我对齐模块目标、产出和验收标准。
```

> 为什么每个模块要新窗口：vLLM 的上下文太长，一个窗口连续做 5 个模块必然被压缩/遗忘。
> 模块文档 + `PROGRESS.md` 就是跨窗口的“记忆”。

---

## 仓库结构

```text
nanovllm/
├── README.md                 # 本文件
├── ROADMAP.md                # 13 个模块、依赖图、里程碑
├── PROGRESS.md               # 跨窗口进度（每个模块结束更新）
├── docs/
│   ├── 00_architecture.md    # 总图 + 冻结接口（先读）
│   ├── M00_contracts.md
│   ├── M01_model.md
│   └── ...                   # 每个模块一篇
├── nanovllm/                 # 你要实现的包
├── tests/                    # M00 测试；后续模块按文档增加
├── solutions/m00/            # 只有 M00 给参考解
├── scripts/
│   ├── check_model_parity.py # M01
│   ├── download_model.py     # 可选
│   └── verify_refs.py        # 检查文档里的 vLLM 行号
└── benchmarks/               # M06/M10/M12
```

## 模块地图

| # | 模块 | 文档 | 你会得到 |
|---|---|---|---|
| M00 | 全景与契约 | [docs/M00_contracts.md](docs/M00_contracts.md) | `Request` / `SchedulerOutput` / 块哈希 |
| M01 | Qwen3 前向 | [docs/M01_model.md](docs/M01_model.md) | 纯 PyTorch Qwen3 + 权重加载 |
| M02 | 朴素 KV + 生成 | [docs/M02_naive_kv.md](docs/M02_naive_kv.md) | 第一次端到端生成文本 |
| M03 | Paged KV | [docs/M03_paged_kv.md](docs/M03_paged_kv.md) | BlockPool / KVCacheManager / BlockTable |
| M04 | Paged Attention | [docs/M04_paged_attention.md](docs/M04_paged_attention.md) | naive + Triton 分页注意力 |
| M05 | Scheduler | [docs/M05_scheduler.md](docs/M05_scheduler.md) | token 预算 / chunked prefill / 抢占 |
| M06 | Engine 端到端 | [docs/M06_engine.md](docs/M06_engine.md) | 连续批处理 + `LLM.generate` |
| M07 | Prefix Caching | [docs/M07_prefix_caching.md](docs/M07_prefix_caching.md) | 前缀命中 / LRU 驱逐 |
| M08 | 采样与输出 | [docs/M08_sampling_output.md](docs/M08_sampling_output.md) | 完整采样 / logprobs / 流式 |
| M09 | 服务化 | [docs/M09_serving.md](docs/M09_serving.md) | OpenAI 兼容 HTTP/SSE |
| M10 | 性能 | [docs/M10_cuda_graph.md](docs/M10_cuda_graph.md) | decode CUDA graph + benchmark |
| M11 | 多进程 EngineCore | [docs/M11_multiprocess.md](docs/M11_multiprocess.md) | 前端/引擎分进程 |
| M12 | Capstone | [docs/M12_capstone.md](docs/M12_capstone.md) | 对拍报告 + 代码地图 |

完整路线、依赖图、时间估算见 [ROADMAP.md](ROADMAP.md)。

---

## 学习规则（很重要）

1. **先读文档，再读代码，最后写代码**。不要一上来让 agent 给你完整实现。
2. **一次只做一个 TODO**。写 5 行、跑一次测试，比写 200 行再 debug 快。
3. **卡住 30 分钟再看参考解**（只有 M00 有）；看完要还原成 TODO 自己再写一遍。
4. **每个模块一个 commit**。commit message 用 `MXX: <一句话>`。
5. **更新 `PROGRESS.md`**：完成了什么、哪里和 vLLM 不一样、下一个模块的入口。
6. **不要改冻结接口**。要改先改 `docs/00_architecture.md` §6，并在 `PROGRESS.md` 记录。
7. **性能测试必须带正确性断言**。快但错，等于没做。

---

## 版本锚点

- 唯一引用源：`/home/dictator/ai_infra/target/vllm-0.32.0/`
- 引用格式：`vllm/v1/request.py:60 · class Request`
- 行号漂移时用 `grep -n "class Request"` 重定位；**符号优先，行号第二**。
- 本机 `/home/dictator/ai_infra/target/vllm-0.32.0` 的 git worktree 元数据不完整（父仓库被移除），无法用 tag 校验；
  所以文档里同时给符号和行号。

## 和 `mini-v1/` 的关系

工作区里有一个更早的 `mini-v1/` 草稿（7 个里程碑，M0 已提交但工作区文件被删）。
它的一些思路（契约层、块哈希）和本课程一致，但粒度过粗，且部分 vLLM 行号引用与当前
`v0.32.0` 树不符（例如 `GPUModelRunner.execute_model` 实际在 3940 行，不是 1719 行）。

**本课程以 `nanovllm/` 和本目录的文档为准**；`mini-v1/` 保留作参考，不需要合并。
如果你已经做过它的 M0，可以快速过一遍 M00，但建议仍然跑一遍本课程的测试，确认接口一致。

---

## 常见问题

**Q: 为什么不直接读 vLLM 的 `Scheduler`？**
A: 它 3400 行，其中 80% 是多模态/connector/DP/PP/spec/structured output。主路径只有
约 200 行。模块文档会帮你把主路径挑出来。

**Q: 为什么用 Qwen3-0.6B？**
A: 单卡 16GB 能跑；结构现代（GQA、QK-Norm、RoPE、SwiGLU、tied embedding），
但不过分复杂；28 层、hidden 1024、8 个 KV head，KV cache 足够大，能观察分页效果。

**Q: 可以换模型吗？**
A: M01 的模型代码是 Qwen3 专用。换 Llama 需要改 RoPE/attention 细节；换更大的 Qwen3
只需要改 config。课程只保证 Qwen3-0.6B。

**Q: 一定要有 GPU 吗？**
A: M00/M05 可以纯 CPU；M01 之后需要 GPU（bf16 + Triton）。没有 GPU 可以只读文档，
或用 CPU 跑一个极小的随机模型验证逻辑。

**Q: 为什么不装 vLLM 做对拍？**
A: vLLM 0.32.0 要求 torch 2.13，会升级/破坏当前 torch 2.11 + CUDA 13 环境。
M12 如果要跑，建议单独 venv。

---

## 现在开始

```bash
cd /home/dictator/ai_infra/nanovllm
python -m pytest tests -q      # 看失败列表
cat docs/M00_contracts.md
```

祝你玩得开心。卡住不是问题，卡住说明你在读真代码。
