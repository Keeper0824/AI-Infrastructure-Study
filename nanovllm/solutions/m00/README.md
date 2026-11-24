# M00 参考解

> 卡住 30 分钟以上再来看。先自己做，做完再对答案。

## 用法

```bash
cd /home/dictator/ai_infra/nanovllm
cp -r solutions/m00/nanovllm/* nanovllm/
python -m pytest tests -q
```

对完答案后，建议把 `nanovllm/` 下的实现还原成 TODO 版，自己再写一遍：

```bash
git checkout -- nanovllm/request.py \
  nanovllm/utils/hashing.py \
  nanovllm/sched/output.py
```

（如果你已经把 TODO 版提交了；没有提交就手动删掉参考实现。）

## 三处关键点

1. `hash_block_tokens` 里 `parent = NONE_HASH if parent is None else parent`。
   `NONE_HASH` 是 32 字节**非空** bytes，所以它和“没有 parent”在类型上可区分。

2. `get_request_block_hasher` 的起点是 `len(request.block_hashes)`，不是 0。
   这个“增量”是幂等的全部秘密：算过的块不会重算，`extend` 也不会重复追加。

3. `validate` 用 `assert` 而不是抛 `ValueError`：和 vLLM 一致（V1 里大量用 assert 做契约校验）。
   如果你想让它在生产里可关闭，可以加 `-O` 运行，但这不是本课程的重点。

## 常见错误对照

| 现象 | 原因 |
|---|---|
| `test_none_parent_uses_none_hash` 失败 | 把 `None` 直接传给了 `hash_function`，没有替换成 `NONE_HASH` |
| `test_hasher_is_incremental_and_idempotent` 失败 | hasher 从 0 开始，重复追加 |
| `test_hasher_chains_parent` 失败 | 每块的 parent 都用了 `None`，没有用上一块的哈希 |
| `test_partial_block_not_hashed_then_hashed_when_full` 失败 | 用了 `ceil` 而不是 `//` |
| `test_cached_fields_must_be_aligned` 没抛异常 | `validate` 只检查了 total，没有检查四个 list 等长 |
