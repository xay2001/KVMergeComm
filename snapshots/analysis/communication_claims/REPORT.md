# F3b / M1 / M2 离线分析

## 完整性

- Matched-budget runs：1568/1568
- F3b 聚合行：432
- RTT 假设：每个方法均按一次请求-响应 RTT；序列化时间使用现有 wire payload。
- Bootstrap：paired percentile 95% CI；分层宏平均保持每个 pair-task 等权。

## M1 matched-budget 点估计

- B-ReKV vs ReKV: Δ=+0.003587, W/L/T=28/28/0
- B-ReKV vs ValueNorm/Evict: Δ=+0.092265, W/L/T=51/5/0
- B-ReKV vs Random: Δ=+0.126680, W/L/T=54/2/0

## M2 paired bootstrap

- B-ReKV - ReKV (56-cell stratified macro): Δ=+0.003587, 95% CI [-0.000712, +0.007895], P(Δ>0)=0.9500
- B-ReKV - ValueNorm/Evict (56-cell stratified macro): Δ=+0.092265, 95% CI [+0.085851, +0.098662], P(Δ>0)=1.0000
- B-ReKV - Random (56-cell stratified macro): Δ=+0.126680, 95% CI [+0.120289, +0.133357], P(Δ>0)=1.0000
- NLD - B-ReKV (9-cell stratified macro): Δ=-0.013333, 95% CI [-0.064444, +0.037778], P(Δ>0)=0.3036
- NLD - ReKV (9-cell stratified macro): Δ=-0.048889, 95% CI [-0.097778, +0.000000], P(Δ>0)=0.0257

## 图

- `/home/xay/KVMergeComm/snapshots/analysis/communication_claims/figures/f3b_latency_curves.png`
- `/home/xay/KVMergeComm/snapshots/analysis/communication_claims/figures/f3b_latency_curves.pdf`
- `/home/xay/KVMergeComm/snapshots/analysis/communication_claims/figures/m1_rekv_pareto_macro.png`
- `/home/xay/KVMergeComm/snapshots/analysis/communication_claims/figures/m1_rekv_pareto_macro.pdf`
