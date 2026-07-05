# B-ReKV-S shift_back 诊断

## 现象

`scripts/run_gpu7_mechanism_extended_full_queue.sh` 在 positional coherence 阶段中断于：

`snapshots/mechanism/logs/gpu7_mechanism_extended_full_0703_1144.log`

已完成 countries 的三项：

- ReKV normal
- ReKV-S / shift_back
- B-ReKV normal

失败项是 B-ReKV-S：`--shift_back` + `--budget_mode coverage`。

## 根因判断

报错位置：

- `models.py:get_short_past_key_values`
- 断言：`assert len(lengths) <= 2`

当前 shift_back forward 会先把压缩后的 `past_key_values` 裁成一个 `short_past_key_values`，用于构建短 cache 的 position / causal mask。这个实现隐含假设：压缩后的 KV cache 最多只有两个不同长度。

B-ReKV 使用 per-sample coverage budget，并且保留 sink/recent token。不同层在 coverage selection 后可能出现多个不同的 KV 长度，因此 `lengths` 的种类数超过 2，触发断言。ReKV-S 的 fixed ratio 没有触发，是因为它的压缩长度更规则。

## 当前绕开方案

已经修改 `scripts/run_gpu7_mechanism_extended_full_queue.sh`：

- 新增 `RUN_BREKV_SHIFT=${RUN_BREKV_SHIFT:-0}`
- positional coherence 默认只跑 ReKV normal / ReKV-S / B-ReKV normal
- 默认跳过 B-ReKV-S，避免队列在 Table 6 之前中断
- 如需重试 B-ReKV-S，可显式设置 `RUN_BREKV_SHIFT=1`

推荐续跑命令：

```bash
RUN_SINK_RECENT=0 RUN_POSITIONAL=1 RUN_BREKV_SHIFT=0 RUN_TABLE6=1 GPU=7 \
  bash scripts/run_gpu7_mechanism_extended_full_queue.sh
```

## 论文处理建议

短期不要强行修 `models.py`，因为 shift_back 的 mask/position 逻辑和动态长度 cache 绑定较深，错误修复不充分可能引入更隐蔽的结果问题。

当前最稳妥写法：

- Table 11 报告 ReKV-S 与 normal 的对比，说明 positional coherence 对 receiver-aware KV communication 仍然重要。
- B-ReKV-S 标记为实现限制或放到未来工作，不作为主结论。
- Table 6 extended tasks 可以先继续跑，不被 B-ReKV-S 阻塞。
