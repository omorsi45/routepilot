# failure_analysis

Run `20261031T071751Z-f0429c6`.

**Labels:** costs are the recorded costs of each model on each item; latency is not used anywhere in this analysis; the learned point is the one the headline reports: lambda tuned on the calibration split; a loss is an item Best Single got right and the routed model got wrong; a gain is the reverse; the test split is read once per replay; nothing is chosen on it; no prompt text is in this report.

## Overall, test split

| quantity | value |
| --- | --- |
| Best Single | gemini-2.5-pro |
| learned point | dual, lambda 6.721045792801306 |
| artifact version | dea4486d7575 |
| items (test split) | 1735 |
| Best Single success / cost USD | 0.5274 / 0.0670642 |
| routed success / cost USD | 0.4818 / 0.00438797 |
| success delta [95% CI] | -0.0455 [-0.0697, -0.0225] |
| cost reduction [95% CI] | 0.9346 [0.9250, 0.9432] |
| both right / both wrong | 663 / 647 |
| losses / gains / net | 252 / 173 / -79 |

## By dataset (worst success delta first)

| dataset | n | Best Single success | routed success | success delta | Best Single cost USD | routed cost USD | cost reduction | losses | gains | net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc-agi | 78 | 0.461538 | 0.25641 | -0.205128 | 0.199776 | 0.00310211 | 0.984472 | 20 | 4 | -16 |
| hle | 432 | 0.215278 | 0.0902778 | -0.125 | 0.124528 | 0.0053521 | 0.957021 | 70 | 16 | -54 |
| gpqa | 33 | 0.818182 | 0.69697 | -0.121212 | 0.105842 | 0.00919564 | 0.913119 | 7 | 3 | -4 |
| livecodebench | 205 | 0.809756 | 0.702439 | -0.107317 | 0.137131 | 0.00721509 | 0.947385 | 32 | 10 | -22 |
| aime | 14 | 0.714286 | 0.642857 | -0.0714286 | 0.175073 | 0.0079499 | 0.954591 | 2 | 1 | -1 |
| arenahard_math | 58 | 0.793103 | 0.758621 | -0.0344828 | 0.055223 | 0.0111255 | 0.798534 | 8 | 6 | -2 |
| arenahard_creative_writing | 51 | 0.960784 | 0.980392 | 0.0196078 | 0.0320819 | 0.00229422 | 0.928489 | 1 | 2 | 1 |
| simpleqa | 864 | 0.564815 | 0.586806 | 0.0219907 | 0.00935572 | 0.00278116 | 0.702732 | 112 | 131 | 19 |

## By the model the policy chose (worst success delta first)

| chosen model | n | Best Single success | routed success | success delta | Best Single cost USD | routed cost USD | cost reduction | losses | gains | net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-2.5-flash | 12 | 0.833333 | 0.333333 | -0.5 | 0.132929 | 0.00508849 | 0.96172 | 6 | 0 | -6 |
| claude-sonnet-4 | 3 | 0.666667 | 0.333333 | -0.333333 | 0.138306 | 0.015467 | 0.888168 | 1 | 0 | -1 |
| intern-s1 | 57 | 0.333333 | 0.122807 | -0.210526 | 0.114825 | 0.00522351 | 0.954509 | 14 | 2 | -12 |
| deepseek-v3-0324 | 65 | 0.584615 | 0.430769 | -0.153846 | 0.131522 | 0.00122358 | 0.990697 | 12 | 2 | -10 |
| kimi-k2-0905 | 150 | 0.633333 | 0.5 | -0.133333 | 0.122631 | 0.00215593 | 0.982419 | 29 | 9 | -20 |
| gpt-5-chat | 97 | 0.618557 | 0.494845 | -0.123711 | 0.0195079 | 0.00161014 | 0.917462 | 18 | 6 | -12 |
| glm-4.6 | 33 | 0.393939 | 0.333333 | -0.0606061 | 0.107618 | 0.0183605 | 0.829391 | 3 | 1 | -2 |
| qwen3-235b-a22b-thinking-2507 | 26 | 0.5 | 0.461538 | -0.0384615 | 0.150614 | 0.0110646 | 0.926537 | 3 | 2 | -1 |
| deepseek-r1-0528 | 62 | 0.451613 | 0.435484 | -0.016129 | 0.105577 | 0.0212285 | 0.798929 | 6 | 5 | -1 |
| qwen3-235b-a22b-2507 | 964 | 0.501037 | 0.485477 | -0.0155602 | 0.0620901 | 0.000950501 | 0.984692 | 146 | 131 | -15 |
| gemini-2.5-pro | 140 | 0.55 | 0.55 | 0 | 0.0107413 | 0.0107413 | 0 | 0 | 0 | 0 |
| gpt-5 | 126 | 0.611111 | 0.619048 | 0.00793651 | 0.0285209 | 0.0160239 | 0.43817 | 14 | 15 | 1 |

## Out of distribution or not

| request | n | Best Single success | routed success | success delta | Best Single cost USD | routed cost USD | cost reduction | losses | gains | net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in distribution | 1713 | 0.522475 | 0.476941 | -0.0455342 | 0.0675489 | 0.00443719 | 0.934311 | 249 | 171 | -78 |
| out of distribution | 22 | 0.909091 | 0.863636 | -0.0454545 | 0.0293266 | 0.000555255 | 0.981066 | 3 | 2 | -1 |

## The 20 largest losses: Best Single right, routed model wrong

| rank | item_id | dataset | chosen |
| --- | --- | --- | --- |
| 1 | arenahard_math/299 | arenahard_math | qwen3-235b-a22b-2507 |
| 2 | arenahard_math/411 | arenahard_math | qwen3-235b-a22b-2507 |
| 3 | livecodebench/491 | livecodebench | gemini-2.5-flash |
| 4 | livecodebench/862 | livecodebench | kimi-k2-0905 |
| 5 | simpleqa/2073 | simpleqa | qwen3-235b-a22b-2507 |
| 6 | simpleqa/2078 | simpleqa | qwen3-235b-a22b-2507 |
| 7 | aime/30 | aime | gpt-5 |
| 8 | livecodebench/435 | livecodebench | gpt-5 |
| 9 | livecodebench/1006 | livecodebench | deepseek-v3-0324 |
| 10 | livecodebench/826 | livecodebench | deepseek-v3-0324 |
| 11 | livecodebench/884 | livecodebench | deepseek-v3-0324 |
| 12 | livecodebench/912 | livecodebench | deepseek-v3-0324 |
| 13 | gpqa/141 | gpqa | kimi-k2-0905 |
| 14 | gpqa/193 | gpqa | kimi-k2-0905 |
| 15 | livecodebench/179 | livecodebench | kimi-k2-0905 |
| 16 | livecodebench/823 | livecodebench | kimi-k2-0905 |
| 17 | gpqa/106 | gpqa | kimi-k2-0905 |
| 18 | gpqa/94 | gpqa | kimi-k2-0905 |
| 19 | livecodebench/1009 | livecodebench | kimi-k2-0905 |
| 20 | livecodebench/146 | livecodebench | kimi-k2-0905 |
