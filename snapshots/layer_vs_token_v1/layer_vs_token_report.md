# Layer-vs-Token same-implementation comparison


## hotpotqa

- skyline = 0.7, full_kv bytes = 18.39 MB

| method | frac | n | score | recovered skyline | total bytes (MB) | A->B bytes (MB) | budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full KV |  | 200 | 0.7350 | 105.0% | 18.39 | 18.39 | 1.0 |
| KVComm layer (query-free) | 0.1 | 200 | 0.2500 | 35.7% | 2.41 | 2.41 | 0.132 |
| KVComm layer (query-free) | 0.3 | 200 | 0.3350 | 47.9% | 5.27 | 5.27 | 0.287 |
| KVComm layer (query-free) | 0.5 | 200 | 0.6350 | 90.7% | 9.26 | 9.26 | 0.504 |
| Random layer | 0.1 | 200 | 0.0100 | 1.4% | 2.41 | 2.41 | 0.132 |
| Random layer | 0.3 | 200 | 0.2300 | 32.9% | 5.27 | 5.27 | 0.287 |
| Random layer | 0.5 | 200 | 0.5850 | 83.6% | 9.26 | 9.26 | 0.504 |
| Receiver-aware layer | 0.1 | 200 | 0.1900 | 27.1% | 4.51 | 2.41 | 0.132 |
| Receiver-aware layer | 0.3 | 200 | 0.2900 | 41.4% | 7.94 | 5.84 | 0.318 |
| Receiver-aware layer | 0.5 | 200 | 0.3050 | 43.6% | 11.93 | 9.83 | 0.535 |
| Query-free token (value-norm) | 0.1 | 200 | 0.1500 | 21.4% | 2.40 | 2.40 | 0.132 |
| Query-free token (value-norm) | 0.3 | 200 | 0.5800 | 82.9% | 5.92 | 5.92 | 0.322 |
| Query-free token (value-norm) | 0.5 | 200 | 0.6800 | 97.1% | 9.49 | 9.49 | 0.516 |
| ReKV (receiver token) | 0.1 | 200 | 0.1950 | 27.9% | 4.50 | 2.40 | 0.132 |
| ReKV (receiver token) | 0.3 | 200 | 0.5900 | 84.3% | 8.02 | 5.92 | 0.322 |
| ReKV (receiver token) | 0.5 | 200 | 0.7150 | 102.1% | 11.59 | 9.49 | 0.516 |
| ReKV shuffled query | 0.1 | 200 | 0.1600 | 22.9% | 4.50 | 2.40 | 0.132 |
| ReKV shuffled query | 0.3 | 200 | 0.4250 | 60.7% | 8.02 | 5.92 | 0.322 |
| ReKV shuffled query | 0.5 | 200 | 0.6050 | 86.4% | 11.59 | 9.49 | 0.516 |

## multifieldqa_en

- skyline = 0.5655, full_kv bytes = 913.84 MB

| method | frac | n | score | recovered skyline | total bytes (MB) | A->B bytes (MB) | budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full KV |  | 145 | 0.5586 | 98.8% | 913.84 | 913.84 | 1.0 |
| KVComm layer (query-free) | 0.1 | 145 | 0.0897 | 15.9% | 85.79 | 85.79 | 0.094 |
| KVComm layer (query-free) | 0.3 | 145 | 0.1793 | 31.7% | 257.11 | 257.11 | 0.281 |
| KVComm layer (query-free) | 0.5 | 145 | 0.4759 | 84.1% | 456.99 | 456.99 | 0.5 |
| Random layer | 0.1 | 145 | 0.0414 | 7.3% | 114.35 | 114.35 | 0.125 |
| Random layer | 0.3 | 145 | 0.0621 | 11.0% | 257.11 | 257.11 | 0.281 |
| Random layer | 0.5 | 145 | 0.2621 | 46.3% | 456.99 | 456.99 | 0.5 |
| Receiver-aware layer | 0.1 | 145 | 0.0414 | 7.3% | 116.44 | 114.35 | 0.125 |
| Receiver-aware layer | 0.3 | 145 | 0.0345 | 6.1% | 287.76 | 285.67 | 0.313 |
| Receiver-aware layer | 0.5 | 145 | 0.0897 | 15.9% | 487.64 | 485.54 | 0.531 |
| Query-free token (value-norm) | 0.1 | 145 | 0.1724 | 30.5% | 117.09 | 117.09 | 0.128 |
| Query-free token (value-norm) | 0.3 | 145 | 0.3310 | 58.5% | 294.14 | 294.14 | 0.322 |
| Query-free token (value-norm) | 0.5 | 145 | 0.3724 | 65.9% | 471.20 | 471.20 | 0.516 |
| ReKV (receiver token) | 0.1 | 145 | 0.3586 | 63.4% | 119.18 | 117.09 | 0.128 |
| ReKV (receiver token) | 0.3 | 145 | 0.4621 | 81.7% | 296.24 | 294.14 | 0.322 |
| ReKV (receiver token) | 0.5 | 145 | 0.4966 | 87.8% | 473.30 | 471.20 | 0.516 |
| ReKV shuffled query | 0.1 | 145 | 0.1862 | 32.9% | 119.18 | 117.09 | 0.128 |
| ReKV shuffled query | 0.3 | 145 | 0.3379 | 59.8% | 296.24 | 294.14 | 0.322 |
| ReKV shuffled query | 0.5 | 145 | 0.4759 | 84.1% | 473.30 | 471.20 | 0.516 |

## musique

- skyline = 0.535, full_kv bytes = 48.3 MB

| method | frac | n | score | recovered skyline | total bytes (MB) | A->B bytes (MB) | budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full KV |  | 200 | 0.5800 | 108.4% | 48.30 | 48.30 | 1.0 |
| KVComm layer (query-free) | 0.1 | 200 | 0.0600 | 11.2% | 6.15 | 6.15 | 0.128 |
| KVComm layer (query-free) | 0.3 | 200 | 0.1200 | 22.4% | 13.68 | 13.68 | 0.284 |
| KVComm layer (query-free) | 0.5 | 200 | 0.3300 | 61.7% | 24.21 | 24.21 | 0.502 |
| Random layer | 0.1 | 200 | 0.0050 | 0.9% | 6.15 | 6.15 | 0.128 |
| Random layer | 0.3 | 200 | 0.0400 | 7.5% | 13.68 | 13.68 | 0.284 |
| Random layer | 0.5 | 200 | 0.3150 | 58.9% | 24.21 | 24.21 | 0.502 |
| Receiver-aware layer | 0.1 | 200 | 0.0100 | 1.9% | 8.25 | 6.15 | 0.128 |
| Receiver-aware layer | 0.3 | 200 | 0.0500 | 9.3% | 17.28 | 15.18 | 0.315 |
| Receiver-aware layer | 0.5 | 200 | 0.0950 | 17.8% | 27.82 | 25.72 | 0.533 |
| Query-free token (value-norm) | 0.1 | 200 | 0.1200 | 22.4% | 6.19 | 6.19 | 0.128 |
| Query-free token (value-norm) | 0.3 | 200 | 0.3600 | 67.3% | 15.55 | 15.55 | 0.322 |
| Query-free token (value-norm) | 0.5 | 200 | 0.4350 | 81.3% | 24.90 | 24.90 | 0.516 |
| ReKV (receiver token) | 0.1 | 200 | 0.1500 | 28.0% | 8.29 | 6.19 | 0.128 |
| ReKV (receiver token) | 0.3 | 200 | 0.3600 | 67.3% | 17.64 | 15.55 | 0.322 |
| ReKV (receiver token) | 0.5 | 200 | 0.4700 | 87.9% | 27.00 | 24.90 | 0.516 |
| ReKV shuffled query | 0.1 | 200 | 0.0750 | 14.0% | 8.29 | 6.19 | 0.128 |
| ReKV shuffled query | 0.3 | 200 | 0.2500 | 46.7% | 17.64 | 15.55 | 0.322 |
| ReKV shuffled query | 0.5 | 200 | 0.3600 | 67.3% | 27.00 | 24.90 | 0.516 |

## Pareto AUC (normalized bytes)

| task | kvcomm | random_layer | recv_layer | evict | rekv |
|---|---:|---:|---:|---:|---:|
| hotpotqa | 0.4048 | 0.2877 | 0.271 | 0.4984 | 0.5233 |
| multifieldqa_en | 0.2385 | 0.1161 | 0.0509 | 0.3017 | 0.4448 |
| musique | 0.1688 | 0.1129 | 0.0529 | 0.3187 | 0.335 |

## Paired differences (bootstrap 95% CI)

| task | A vs B | frac | mean diff | 95% CI | win/loss/tie | bytes A/B (MB) |
|---|---|---:|---:|---|---|---|
| hotpotqa | rekv vs recv_layer | 0.1 | +0.0050 | [-0.0600, +0.0700] | 23/22/155 | 4.502/4.512 |
| hotpotqa | rekv vs recv_layer | 0.3 | +0.3000 | [+0.2250, +0.3750] | 69/9/122 | 8.021/7.936 |
| hotpotqa | rekv vs recv_layer | 0.5 | +0.4100 | [+0.3400, +0.4850] | 88/6/106 | 11.585/11.93 |
| hotpotqa | rekv vs kvcomm | 0.1 | -0.0550 | [-0.1200, +0.0100] | 16/27/157 | 4.502/2.414 |
| hotpotqa | rekv vs kvcomm | 0.3 | +0.2550 | [+0.1750, +0.3350] | 65/14/121 | 8.021/5.268 |
| hotpotqa | rekv vs kvcomm | 0.5 | +0.0800 | [+0.0200, +0.1450] | 29/13/158 | 11.585/9.262 |
| hotpotqa | rekv vs evict | 0.1 | +0.0450 | [+0.0000, +0.0900] | 14/5/181 | 4.502/2.404 |
| hotpotqa | rekv vs evict | 0.3 | +0.0100 | [-0.0650, +0.0850] | 28/26/146 | 8.021/5.923 |
| hotpotqa | rekv vs evict | 0.5 | +0.0350 | [-0.0200, +0.0900] | 18/11/171 | 11.585/9.488 |
| hotpotqa | recv_layer vs kvcomm | 0.1 | -0.0600 | [-0.1150, -0.0050] | 10/22/168 | 4.512/2.414 |
| hotpotqa | recv_layer vs kvcomm | 0.3 | -0.0450 | [-0.1150, +0.0250] | 21/30/149 | 7.936/5.268 |
| hotpotqa | recv_layer vs kvcomm | 0.5 | -0.3300 | [-0.4050, -0.2600] | 5/71/124 | 11.93/9.262 |
| hotpotqa | recv_layer vs random_layer | 0.1 | +0.1800 | [+0.1250, +0.2400] | 38/2/160 | 4.512/2.414 |
| hotpotqa | recv_layer vs random_layer | 0.3 | +0.0600 | [+0.0100, +0.1100] | 19/7/174 | 7.936/5.268 |
| hotpotqa | recv_layer vs random_layer | 0.5 | -0.2800 | [-0.3500, -0.2100] | 6/62/132 | 11.93/9.262 |
| hotpotqa | kvcomm vs random_layer | 0.1 | +0.2400 | [+0.1800, +0.3000] | 49/1/150 | 2.414/2.414 |
| hotpotqa | kvcomm vs random_layer | 0.3 | +0.1050 | [+0.0400, +0.1750] | 33/12/155 | 5.268/5.268 |
| hotpotqa | kvcomm vs random_layer | 0.5 | +0.0500 | [-0.0150, +0.1200] | 31/21/148 | 9.262/9.262 |
| hotpotqa | rekv vs rekv_shuffled | 0.1 | +0.0350 | [+0.0000, +0.0700] | 10/3/187 | 4.502/4.502 |
| hotpotqa | rekv vs rekv_shuffled | 0.3 | +0.1650 | [+0.1000, +0.2350] | 42/9/149 | 8.021/8.021 |
| hotpotqa | rekv vs rekv_shuffled | 0.5 | +0.1100 | [+0.0550, +0.1650] | 28/6/166 | 11.585/11.585 |
| multifieldqa_en | rekv vs recv_layer | 0.1 | +0.3172 | [+0.2276, +0.4069] | 51/5/89 | 119.184/116.444 |
| multifieldqa_en | rekv vs recv_layer | 0.3 | +0.4276 | [+0.3379, +0.5172] | 67/5/73 | 296.237/287.765 |
| multifieldqa_en | rekv vs recv_layer | 0.5 | +0.4069 | [+0.3103, +0.5034] | 66/7/72 | 473.295/487.639 |
| multifieldqa_en | rekv vs kvcomm | 0.1 | +0.2690 | [+0.1793, +0.3586] | 45/6/94 | 119.184/85.792 |
| multifieldqa_en | rekv vs kvcomm | 0.3 | +0.2828 | [+0.2000, +0.3655] | 45/4/96 | 296.237/257.113 |
| multifieldqa_en | rekv vs kvcomm | 0.5 | +0.0207 | [-0.0414, +0.0828] | 12/9/124 | 473.295/456.988 |
| multifieldqa_en | rekv vs evict | 0.1 | +0.1862 | [+0.0966, +0.2690] | 36/9/100 | 119.184/117.086 |
| multifieldqa_en | rekv vs evict | 0.3 | +0.1310 | [+0.0552, +0.2069] | 28/9/108 | 296.237/294.139 |
| multifieldqa_en | rekv vs evict | 0.5 | +0.1241 | [+0.0621, +0.1931] | 23/5/117 | 473.295/471.198 |
| multifieldqa_en | recv_layer vs kvcomm | 0.1 | -0.0483 | [-0.0966, +0.0000] | 3/10/132 | 116.444/85.792 |
| multifieldqa_en | recv_layer vs kvcomm | 0.3 | -0.1448 | [-0.2207, -0.0759] | 5/26/114 | 287.765/257.113 |
| multifieldqa_en | recv_layer vs kvcomm | 0.5 | -0.3862 | [-0.4828, -0.2897] | 7/63/75 | 487.639/456.988 |
| multifieldqa_en | recv_layer vs random_layer | 0.1 | +0.0000 | [-0.0276, +0.0276] | 2/2/141 | 116.444/114.346 |
| multifieldqa_en | recv_layer vs random_layer | 0.3 | -0.0276 | [-0.0690, +0.0138] | 3/7/135 | 287.765/257.113 |
| multifieldqa_en | recv_layer vs random_layer | 0.5 | -0.1724 | [-0.2552, -0.0966] | 6/31/108 | 487.639/456.988 |
| multifieldqa_en | kvcomm vs random_layer | 0.1 | +0.0483 | [+0.0000, +0.0966] | 11/4/130 | 85.792/114.346 |
| multifieldqa_en | kvcomm vs random_layer | 0.3 | +0.1172 | [+0.0414, +0.1862] | 24/7/114 | 257.113/257.113 |
| multifieldqa_en | kvcomm vs random_layer | 0.5 | +0.2138 | [+0.1241, +0.3034] | 40/9/96 | 456.988/456.988 |
| multifieldqa_en | rekv vs rekv_shuffled | 0.1 | +0.1724 | [+0.1034, +0.2483] | 29/4/112 | 119.184/119.184 |
| multifieldqa_en | rekv vs rekv_shuffled | 0.3 | +0.1241 | [+0.0483, +0.2000] | 26/8/111 | 296.237/296.237 |
| multifieldqa_en | rekv vs rekv_shuffled | 0.5 | +0.0207 | [-0.0345, +0.0759] | 10/7/128 | 473.295/473.295 |
| musique | rekv vs recv_layer | 0.1 | +0.1400 | [+0.0900, +0.1900] | 29/1/170 | 8.288/8.25 |
| musique | rekv vs recv_layer | 0.3 | +0.3100 | [+0.2400, +0.3800] | 65/3/132 | 17.644/17.281 |
| musique | rekv vs recv_layer | 0.5 | +0.3750 | [+0.3050, +0.4500] | 79/4/117 | 26.999/27.817 |
| musique | rekv vs kvcomm | 0.1 | +0.0900 | [+0.0450, +0.1350] | 21/3/176 | 8.288/6.152 |
| musique | rekv vs kvcomm | 0.3 | +0.2400 | [+0.1750, +0.3050] | 53/5/142 | 17.644/13.678 |
| musique | rekv vs kvcomm | 0.5 | +0.1400 | [+0.0750, +0.2000] | 36/8/156 | 26.999/24.214 |
| musique | rekv vs evict | 0.1 | +0.0300 | [-0.0200, +0.0800] | 18/12/170 | 8.288/6.19 |
| musique | rekv vs evict | 0.3 | +0.0000 | [-0.0700, +0.0700] | 26/26/148 | 17.644/15.546 |
| musique | rekv vs evict | 0.5 | +0.0350 | [-0.0250, +0.0950] | 21/14/165 | 26.999/24.901 |
| musique | recv_layer vs kvcomm | 0.1 | -0.0500 | [-0.0850, -0.0200] | 1/11/188 | 8.25/6.152 |
| musique | recv_layer vs kvcomm | 0.3 | -0.0700 | [-0.1200, -0.0250] | 5/19/176 | 17.281/13.678 |
| musique | recv_layer vs kvcomm | 0.5 | -0.2350 | [-0.3100, -0.1650] | 8/55/137 | 27.817/24.214 |
| musique | recv_layer vs random_layer | 0.1 | +0.0050 | [+0.0000, +0.0150] | 1/0/199 | 8.25/6.152 |
| musique | recv_layer vs random_layer | 0.3 | +0.0100 | [-0.0100, +0.0300] | 3/1/196 | 17.281/13.678 |
| musique | recv_layer vs random_layer | 0.5 | -0.2200 | [-0.2900, -0.1550] | 5/49/146 | 27.817/24.214 |
| musique | kvcomm vs random_layer | 0.1 | +0.0550 | [+0.0250, +0.0900] | 12/1/187 | 6.152/6.152 |
| musique | kvcomm vs random_layer | 0.3 | +0.0800 | [+0.0350, +0.1300] | 20/4/176 | 13.678/13.678 |
| musique | kvcomm vs random_layer | 0.5 | +0.0150 | [-0.0550, +0.0950] | 31/28/141 | 24.214/24.214 |
| musique | rekv vs rekv_shuffled | 0.1 | +0.0750 | [+0.0250, +0.1200] | 21/6/173 | 8.288/8.288 |
| musique | rekv vs rekv_shuffled | 0.3 | +0.1100 | [+0.0500, +0.1750] | 32/10/158 | 17.644/17.644 |
| musique | rekv vs rekv_shuffled | 0.5 | +0.1100 | [+0.0600, +0.1650] | 27/5/168 | 26.999/26.999 |
