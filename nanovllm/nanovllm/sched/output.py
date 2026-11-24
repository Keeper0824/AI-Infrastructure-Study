"""SchedulerOutput：调度器交给执行器的“本步指令”。

对照 vLLM：
    vllm/v1/core/sched/output.py:41    NewRequestData
    vllm/v1/core/sched/output.py:138   CachedRequestData
    vllm/v1/core/sched/output.py:232   SchedulerOutput

设计意图（vLLM 原注释）：
    "We cache the request's data in each worker process, so that we don't
     need to re-send it every scheduling step."
    "Since the request's data is already cached in the worker processes,
     we only send the diff to minimize the communication cost."

翻译：**新请求发全量（NewRequestData），老请求只发增量（CachedRequestData）。**
这就是为什么 D 个请求跑 1000 步，通信量是 O(D + steps) 而不是 O(D × steps)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanovllm.sampling_params import SamplingParams


@dataclass
class NewRequestData:
    """第一次被调度的请求：发全量，worker 会把它缓存起来。"""

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # 当前请求已经持有的物理块 id；mini 版只有一组 KV cache
    block_ids: list[int]
    num_computed_tokens: int
    num_output_tokens: int = 0


@dataclass
class CachedRequestData:
    """已被调度过的请求：只发 diff。各 list 按 ``req_ids`` 的顺序对齐。"""

    req_ids: list[str]
    # 本步从 WAITING 重新回到 RUNNING 的请求（被抢占过 / 刚做完前缀命中）
    resumed_req_ids: set[str] = field(default_factory=set)
    # 本步新分配的 block ids；None 表示该请求本步没有新块
    new_block_ids: list[list[int] | None] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(req_ids=[])


@dataclass
class SchedulerOutput:
    """一个调度步的全部信息。**不可变**：生成之后 worker 只读。"""

    # 本步第一次被调度的请求（发全量）
    scheduled_new_reqs: list[NewRequestData]
    # 之前调度过的请求（发 diff）
    scheduled_cached_reqs: CachedRequestData
    # req_id -> 本步要算的 token 数
    num_scheduled_tokens: dict[str, int]
    # 必须等于 sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # 投机解码的 token（mini 版恒为空）
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    # 本步完成的请求（通知 worker 释放缓存状态）
    finished_req_ids: set[str] = field(default_factory=set)
    # 本步被抢占的请求
    preempted_req_ids: set[str] = field(default_factory=set)

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        )

    def validate(self, max_num_batched_tokens: int) -> None:
        """守住 docs/00_architecture.md §6.3 的 7 条不变量。

        用 ``assert``（与 vLLM 一致），失败时给出能看懂的信息。
        """
        # TODO(M00-6):
        #   1) total == sum(num_scheduled_tokens.values())
        #   2) total <= max_num_batched_tokens
        #   3) 每个 num_scheduled_tokens > 0
        #   4) finished_req_ids 与 num_scheduled_tokens 不相交
        #   5) scheduled_new_reqs 的 id 都在 num_scheduled_tokens 里
        #   6) new / cached 的 id 不重复
        #   7) cached 的四个 list 等长，且 id 都在 num_scheduled_tokens 里
        assert self.total_num_scheduled_tokens == sum(
            self.num_scheduled_tokens.values()
        ), "本step需要计算的token与逐req_id请求的不同"
        assert (
            self.total_num_scheduled_tokens <= max_num_batched_tokens
        ), "本step需要计算的token超过最大批处理数量"

        def check_num_scheduled_tokens_legal(temp: dict[str, int]):
            for cnt in temp.values():
                if cnt <= 0:
                    return False
            return True

        assert check_num_scheduled_tokens_legal(
            self.num_scheduled_tokens
        ), "计划计算的token小于等于0，无需计算"

        scheduled_ids = self.num_scheduled_tokens.keys()
        new_ids = set(s.req_id for s in self.scheduled_new_reqs)
        cached_ids = self.scheduled_cached_reqs.req_ids

        assert scheduled_ids.isdisjoint(
            self.finished_req_ids
        ), "已经完成的请求仍然出现在计算中"

        assert new_ids <= set(scheduled_ids), "新送入请求未进入计划计算"

        assert new_ids.isdisjoint(cached_ids), "新请求和已缓存请求重复"

        assert (
            len(self.scheduled_cached_reqs.req_ids)
            == len(self.scheduled_cached_reqs.new_block_ids)
            == len(self.scheduled_cached_reqs.num_computed_tokens)
            == len(self.scheduled_cached_reqs.num_output_tokens)
        ), "缓存的请求发生错误，list长度不一致"

        assert set(cached_ids) <= set(scheduled_ids), "缓存的请求未出现在调度队列中"
        # raise NotImplementedError("M00-6: implement SchedulerOutput.validate()")
