# Dataset summary

Run `20261031T064740Z-d02bb26`. Dataset SHA-256 `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`.

Counts, rates and item IDs only: no prompt text. Prices are not list prices: they are fitted from recorded costs (non-negative least squares on prompt and completion tokens, rows with a non-zero recorded cost).

## Held-out datasets

Chosen before training by `sha256(name + salt)` order, at most one per domain, salt `routepilot-split-v1`.

| dataset | domain |
|---|---|
| arenahard_coding | code |
| livemathbench | math |
| mmlupro | knowledge |

## Items per dataset and split

| dataset | train | calib | test | heldout | total |
|---|---|---|---|---|---|
| aime | 36 | 10 | 14 | 0 | 60 |
| arc-agi | 251 | 71 | 78 | 0 | 400 |
| arenahard_coding | 0 | 0 | 0 | 253 | 253 |
| arenahard_creative_writing | 151 | 48 | 51 | 0 | 250 |
| arenahard_math | 131 | 58 | 58 | 0 | 247 |
| gpqa | 131 | 34 | 33 | 0 | 198 |
| hle | 1271 | 454 | 433 | 0 | 2158 |
| livecodebench | 618 | 232 | 205 | 0 | 1055 |
| livemathbench | 0 | 0 | 0 | 121 | 121 |
| mmlupro | 0 | 0 | 0 | 3000 | 3000 |
| simpleqa | 2596 | 866 | 864 | 0 | 4326 |
| total | 5185 | 1773 | 1736 | 3374 | 12068 |

## Pool `flagship`

12067 items with a record from every one of 12 models, over 11 datasets: aime, arc-agi, arenahard_coding, arenahard_creative_writing, arenahard_math, gpqa, hle, livecodebench, livemathbench, mmlupro, simpleqa. Items per split: train 5185, calib 1773, test 1735, heldout 3374.

- Best Single (highest train+calib success): `gemini-2.5-pro`
- Cheapest (lowest mean cost per item on train+calib, as the replay charges it): `deepseek-v3-0324`

### Success per split (pool `flagship`)

| model | rank | train | calib | test | heldout |
|---|---|---|---|---|---|
| gemini-2.5-pro | 12 | 0.511 | 0.501 | 0.527 | 0.845 |
| gpt-5 | 11 | 0.495 | 0.514 | 0.516 | 0.868 |
| qwen3-235b-a22b-2507 | 10 | 0.444 | 0.443 | 0.459 | 0.825 |
| qwen3-235b-a22b-thinking-2507 | 9 | 0.433 | 0.440 | 0.428 | 0.786 |
| gpt-5-chat | 8 | 0.360 | 0.369 | 0.368 | 0.809 |
| deepseek-r1-0528 | 7 | 0.349 | 0.345 | 0.348 | 0.828 |
| glm-4.6 | 6 | 0.311 | 0.320 | 0.333 | 0.797 |
| kimi-k2-0905 | 5 | 0.307 | 0.320 | 0.322 | 0.801 |
| gemini-2.5-flash | 4 | 0.306 | 0.299 | 0.307 | 0.793 |
| deepseek-v3-0324 | 3 | 0.296 | 0.280 | 0.290 | 0.772 |
| intern-s1 | 2 | 0.221 | 0.228 | 0.217 | 0.814 |
| claude-sonnet-4 | 1 | 0.223 | 0.218 | 0.229 | 0.814 |

### Prices fitted from recorded costs (pool `flagship`)

| model | in USD/Mtok | out USD/Mtok | RMSE USD | R^2 | rows fitted | replay cost | flag |
|---|---|---|---|---|---|---|---|
| claude-sonnet-4 | 3.0000 | 15.0000 | 0.000000 | 1.00000 | 12066 | recorded |  |
| deepseek-r1-0528 | 0.5000 | 2.1500 | 0.000000 | 1.00000 | 12067 | recorded |  |
| deepseek-v3-0324 | 0.2500 | 0.8800 | 0.000000 | 1.00000 | 12067 | recorded |  |
| gemini-2.5-flash | 0.1859 | 2.4766 | 0.002721 | 0.99117 | 12035 | recorded |  |
| gemini-2.5-pro | 1.1582 | 10.0019 | 0.000248 | 0.99999 | 11866 | recorded |  |
| glm-4.6 | 0.5639 | 2.2003 | 0.000212 | 0.99997 | 12038 | recorded |  |
| gpt-5 | 1.1357 | 10.0039 | 0.000327 | 0.99997 | 11529 | recorded |  |
| gpt-5-chat | 1.5039 | 8.8517 | 0.002184 | 0.88128 | 12067 | recorded | R^2 below 0.9 |
| intern-s1 | 0.1800 | 0.5400 | 0.000000 | 1.00000 | 12067 | recorded |  |
| kimi-k2-0905 | 0.4983 | 2.0045 | 0.000095 | 0.99858 | 12065 | recorded |  |
| qwen3-235b-a22b-2507 | 0.0235 | 0.6674 | 0.000594 | 0.88706 | 11868 | recorded | R^2 below 0.9 |
| qwen3-235b-a22b-thinking-2507 | 1.4319 | 1.5360 | 0.004850 | 0.68852 | 10659 | recorded | R^2 below 0.9 |

R^2 range: 0.68852 to 1.00000. Fits with R^2 below 0.9 (still used as fitted): gpt-5-chat, qwen3-235b-a22b-2507, qwen3-235b-a22b-thinking-2507.

## Pool `mixed`

3182 items with a record from every one of 16 models, over 8 datasets: aime, arenahard_coding, arenahard_creative_writing, arenahard_math, gpqa, livecodebench, livemathbench, mmlupro. Items per split: train 1067, calib 382, test 361, heldout 1372.

- Best Single (highest train+calib success): `gpt-5`
- Cheapest (lowest mean cost per item on train+calib, as the replay charges it): `Llama-3.1-Nemotron-Nano-8B-v1`
- Small models added to the flagship pool: `DeepSeek-R1-0528-Qwen3-8B`, `Llama-3.1-Nemotron-Nano-8B-v1`, `NVIDIA-Nemotron-Nano-9B-v2`, `Qwen3-8B`. Their cost in the replay export is recomputed from their prices (fitted from recorded costs) and their recorded token counts, because their recorded cost is zero on some datasets. Flagship models keep their recorded cost.

### Success per split (pool `mixed`)

| model | rank | train | calib | test | heldout |
|---|---|---|---|---|---|
| gpt-5 | 16 | 0.834 | 0.866 | 0.837 | 0.849 |
| gemini-2.5-pro | 15 | 0.828 | 0.804 | 0.825 | 0.815 |
| qwen3-235b-a22b-thinking-2507 | 14 | 0.797 | 0.796 | 0.817 | 0.780 |
| deepseek-r1-0528 | 13 | 0.794 | 0.772 | 0.748 | 0.796 |
| kimi-k2-0905 | 12 | 0.705 | 0.709 | 0.745 | 0.795 |
| qwen3-235b-a22b-2507 | 11 | 0.689 | 0.688 | 0.726 | 0.816 |
| gpt-5-chat | 10 | 0.691 | 0.670 | 0.706 | 0.784 |
| glm-4.6 | 9 | 0.681 | 0.688 | 0.698 | 0.771 |
| deepseek-v3-0324 | 8 | 0.664 | 0.662 | 0.673 | 0.746 |
| gemini-2.5-flash | 7 | 0.632 | 0.657 | 0.626 | 0.758 |
| claude-sonnet-4 | 6 | 0.603 | 0.594 | 0.618 | 0.778 |
| intern-s1 | 5 | 0.601 | 0.597 | 0.579 | 0.801 |
| Qwen3-8B | 4 | 0.584 | 0.558 | 0.568 | 0.617 |
| DeepSeek-R1-0528-Qwen3-8B | 3 | 0.575 | 0.531 | 0.562 | 0.614 |
| NVIDIA-Nemotron-Nano-9B-v2 | 2 | 0.533 | 0.497 | 0.485 | 0.603 |
| Llama-3.1-Nemotron-Nano-8B-v1 | 1 | 0.351 | 0.340 | 0.285 | 0.369 |

### Prices fitted from recorded costs (pool `mixed`)

| model | in USD/Mtok | out USD/Mtok | RMSE USD | R^2 | rows fitted | replay cost | flag |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-0528-Qwen3-8B | 0.0300 | 0.1100 | 0.000000 | 1.00000 | 2432 | fitted |  |
| Llama-3.1-Nemotron-Nano-8B-v1 | 0.0150 | 0.0200 | 0.000000 | 1.00000 | 2432 | fitted |  |
| NVIDIA-Nemotron-Nano-9B-v2 | 0.0400 | 0.1600 | 0.000000 | 1.00000 | 2432 | fitted |  |
| Qwen3-8B | 0.0350 | 0.1380 | 0.000000 | 1.00000 | 2432 | fitted |  |
| claude-sonnet-4 | 3.0000 | 15.0000 | 0.000000 | 1.00000 | 3181 | recorded |  |
| deepseek-r1-0528 | 0.5000 | 2.1500 | 0.000000 | 1.00000 | 3182 | recorded |  |
| deepseek-v3-0324 | 0.2500 | 0.8800 | 0.000000 | 1.00000 | 3182 | recorded |  |
| gemini-2.5-flash | 0.0000 | 2.2684 | 0.005617 | 0.91162 | 3176 | recorded |  |
| gemini-2.5-pro | 1.2500 | 10.0000 | 0.000000 | 1.00000 | 3061 | recorded |  |
| glm-4.6 | 0.0000 | 2.1982 | 0.000345 | 0.99961 | 3177 | recorded |  |
| gpt-5 | 1.2500 | 10.0000 | 0.000000 | 1.00000 | 3164 | recorded |  |
| gpt-5-chat | 3.9382 | 5.9648 | 0.003911 | 0.42323 | 3182 | recorded | R^2 below 0.9 |
| intern-s1 | 0.1800 | 0.5400 | 0.000000 | 1.00000 | 3182 | recorded |  |
| kimi-k2-0905 | 0.4883 | 2.0065 | 0.000209 | 0.99718 | 3180 | recorded |  |
| qwen3-235b-a22b-2507 | 0.0000 | 0.6613 | 0.000225 | 0.97203 | 3174 | recorded |  |
| qwen3-235b-a22b-thinking-2507 | 4.4246 | 1.2810 | 0.006538 | 0.58369 | 2942 | recorded | R^2 below 0.9 |

R^2 range: 0.42323 to 1.00000. Fits with R^2 below 0.9 (still used as fitted): gpt-5-chat, qwen3-235b-a22b-thinking-2507.

## Zero recorded cost (flagship pool)

Records with `cost_usd == 0`, per model and dataset, with the count that also have zero completion tokens in brackets. They are kept as recorded.

| model | aime | arc-agi | arenahard_coding | arenahard_creative_writing | arenahard_math | gpqa | hle | livecodebench | livemathbench | mmlupro | simpleqa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-4 | 1 (1) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| deepseek-r1-0528 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| deepseek-v3-0324 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| gemini-2.5-flash | 1 (1) | 0 (0) | 2 (2) | 1 (1) | 1 (1) | 0 (0) | 20 (20) | 0 (0) | 0 (0) | 5 (5) | 2 (2) |
| gemini-2.5-pro | 0 (0) | 6 (6) | 3 (3) | 2 (2) | 3 (3) | 0 (0) | 48 (48) | 95 (95) | 1 (1) | 33 (33) | 10 (10) |
| glm-4.6 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 2 (2) | 3 (3) | 0 (0) | 6 (6) | 18 (18) |
| gpt-5 | 1 (1) | 4 (4) | 6 (6) | 2 (2) | 3 (3) | 0 (0) | 507 (3) | 0 (0) | 4 (4) | 2 (2) | 9 (4) |
| gpt-5-chat | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| intern-s1 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| kimi-k2-0905 | 1 (1) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 1 (1) | 0 (0) | 0 (0) | 0 (0) |
| qwen3-235b-a22b-2507 | 1 (1) | 42 (42) | 4 (4) | 0 (0) | 1 (1) | 0 (0) | 123 (118) | 0 (0) | 0 (0) | 14 (14) | 14 (14) |
| qwen3-235b-a22b-thinking-2507 | 7 (7) | 232 (232) | 20 (20) | 1 (1) | 39 (39) | 0 (0) | 780 (780) | 117 (117) | 37 (37) | 174 (174) | 1 (1) |

## Negative token counts

10 records have a negative token count. They are kept as recorded.

| item_id | model | prompt_tokens | completion_tokens | cost_usd |
|---|---|---|---|---|
| hle/1144 | gpt-5 | 0 | -56 | 0.0 |
| hle/1265 | gpt-5 | 0 | -48 | 0.0 |
| hle/2000 | gpt-5 | 0 | -25 | 0.0 |
| hle/2072 | gpt-5 | 0 | -54 | 0.0 |
| hle/554 | gpt-5 | 0 | -52 | 0.0 |
| simpleqa/1137 | gpt-5 | 0 | -36 | 0.0 |
| simpleqa/2574 | gpt-5 | 0 | -38 | 0.0 |
| simpleqa/2683 | gpt-5 | 0 | -21 | 0.0 |
| simpleqa/3592 | gpt-5 | 0 | -2 | 0.0 |
| simpleqa/4113 | gpt-5 | 0 | -17 | 0.0 |
