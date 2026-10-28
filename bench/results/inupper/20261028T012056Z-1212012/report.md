# inupper

Run `20261028T012056Z-1212012`.

**Labels:** in_upper is the router's upper bound on prompt tokens, computed from the prompt bytes; prompt_tokens are the counts recorded in the dataset; nothing here is simulated; prompt_overhead_tokens are fitted on train and calib records only, so the test and heldout rows are an out-of-sample check.

## Ratio of recorded prompt tokens to in_upper, per pool

| pool | records | max_ratio | p50 | p99 | p99.9 | count_above_1 | nonpositive_prompt_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | 142926 | 5.09274 | 0.267281 | 1.79798 | 2.5 | 4319 | 1878 |
| mixed | 50521 | 5.09274 | 0.282653 | 0.717219 | 2.00518 | 238 | 391 |

## Per dataset

| pool | dataset | records | max_ratio | p50 | p99 | p99.9 | count_above_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | aime | 709 | 0.648294 | 0.265976 | 0.612299 | 0.645463 | 0 |
| flagship | arc-agi | 4520 | 0.975155 | 0.817429 | 0.966088 | 0.969976 | 0 |
| flagship | arenahard_coding | 3006 | 2.47761 | 0.243893 | 1.22433 | 1.96501 | 54 |
| flagship | arenahard_creative_writing | 2996 | 3.95122 | 0.236181 | 1.87138 | 3.32698 | 97 |
| flagship | arenahard_math | 2920 | 4.20513 | 0.282946 | 1.79995 | 3.54014 | 86 |
| flagship | gpqa | 2376 | 0.759791 | 0.306327 | 0.641222 | 0.707575 | 0 |
| flagship | hle | 24911 | 0.976735 | 0.264851 | 0.660946 | 0.775088 | 0 |
| flagship | livecodebench | 12444 | 0.557663 | 0.292109 | 0.43749 | 0.499363 | 0 |
| flagship | livemathbench | 1414 | 5.09274 | 0.267019 | 0.667209 | 0.718948 | 1 |
| flagship | mmlupro | 35768 | 0.909385 | 0.292911 | 0.712743 | 0.838806 | 0 |
| flagship | simpleqa | 51862 | 3.24 | 0.243902 | 2.09569 | 2.68922 | 4081 |
| mixed | aime | 949 | 0.648294 | 0.267148 | 0.601481 | 0.644503 | 0 |
| mixed | arenahard_coding | 4018 | 2.47761 | 0.243902 | 1.11748 | 1.94362 | 54 |
| mixed | arenahard_creative_writing | 3996 | 3.95122 | 0.239062 | 1.6258 | 3.19673 | 97 |
| mixed | arenahard_math | 3901 | 4.20513 | 0.284848 | 1.50725 | 3.454 | 86 |
| mixed | gpqa | 3168 | 0.759791 | 0.3044 | 0.626061 | 0.69361 | 0 |
| mixed | livecodebench | 16664 | 0.557663 | 0.290825 | 0.434169 | 0.499253 | 0 |
| mixed | livemathbench | 1898 | 5.09274 | 0.269231 | 0.659429 | 0.716558 | 1 |
| mixed | mmlupro | 15927 | 0.902736 | 0.288939 | 0.675418 | 0.820042 | 0 |

## Per model

| pool | model | records | max_ratio | p50 | p99 | p99.9 | count_above_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | claude-sonnet-4 | 12066 | 0.96756 | 0.285714 | 0.877005 | 0.951609 | 0 |
| flagship | deepseek-r1-0528 | 12067 | 0.966088 | 0.23185 | 0.87189 | 0.949611 | 0 |
| flagship | deepseek-v3-0324 | 12067 | 0.966088 | 0.23185 | 0.87189 | 0.949611 | 0 |
| flagship | gemini-2.5-flash | 12035 | 0.967089 | 0.23183 | 0.877547 | 0.951147 | 0 |
| flagship | gemini-2.5-pro | 11866 | 0.967089 | 0.231051 | 0.874773 | 0.950682 | 0 |
| flagship | glm-4.6 | 12038 | 0.966205 | 0.246753 | 0.872398 | 0.9498 | 0 |
| flagship | gpt-5 | 12056 | 5.09274 | 0.252427 | 0.872208 | 0.950148 | 1 |
| flagship | gpt-5-chat | 12067 | 0.966323 | 0.258427 | 0.872803 | 0.949974 | 0 |
| flagship | intern-s1 | 12067 | 4.20513 | 0.656371 | 2.55385 | 3.19608 | 4318 |
| flagship | kimi-k2-0905 | 12065 | 0.96756 | 0.355556 | 0.880935 | 0.951881 | 0 |
| flagship | qwen3-235b-a22b-2507 | 11873 | 0.966382 | 0.271739 | 0.829281 | 0.944752 | 0 |
| flagship | qwen3-235b-a22b-thinking-2507 | 10659 | 0.958895 | 0.276102 | 0.71802 | 0.933179 | 0 |
| mixed | DeepSeek-R1-0528-Qwen3-8B | 3182 | 0.913746 | 0.269713 | 0.484464 | 0.635802 | 0 |
| mixed | Llama-3.1-Nemotron-Nano-8B-v1 | 3182 | 0.817073 | 0.28368 | 0.479572 | 0.625158 | 0 |
| mixed | NVIDIA-Nemotron-Nano-9B-v2 | 3175 | 0.916133 | 0.288927 | 0.518466 | 0.696573 | 0 |
| mixed | Qwen3-8B | 3182 | 0.914289 | 0.277273 | 0.501684 | 0.67207 | 0 |
| mixed | claude-sonnet-4 | 3181 | 0.685185 | 0.30228 | 0.51706 | 0.630885 | 0 |
| mixed | deepseek-r1-0528 | 3182 | 0.650206 | 0.262797 | 0.429963 | 0.514467 | 0 |
| mixed | deepseek-v3-0324 | 3182 | 0.650206 | 0.262797 | 0.429963 | 0.514467 | 0 |
| mixed | gemini-2.5-flash | 3176 | 0.918629 | 0.275282 | 0.492755 | 0.632226 | 0 |
| mixed | gemini-2.5-pro | 3061 | 0.918629 | 0.273171 | 0.493502 | 0.633417 | 0 |
| mixed | glm-4.6 | 3177 | 0.804878 | 0.270525 | 0.460489 | 0.588355 | 0 |
| mixed | gpt-5 | 3181 | 5.09274 | 0.269231 | 0.44569 | 0.542909 | 1 |
| mixed | gpt-5-chat | 3182 | 0.63786 | 0.270646 | 0.44773 | 0.536518 | 0 |
| mixed | intern-s1 | 3182 | 4.20513 | 0.438266 | 2.3452 | 3.55873 | 237 |
| mixed | kimi-k2-0905 | 3180 | 0.944444 | 0.299464 | 0.636551 | 0.892923 | 0 |
| mixed | qwen3-235b-a22b-2507 | 3174 | 0.914289 | 0.279365 | 0.501943 | 0.672344 | 0 |
| mixed | qwen3-235b-a22b-thinking-2507 | 2942 | 0.75 | 0.279482 | 0.493216 | 0.644838 | 0 |

## Records above 1 (worst first)

| pool | item_id | dataset | model | prompt_tokens | in_upper | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| flagship | livemathbench/7 | livemathbench | gpt-5 | 3295 | 647 | 5.09274 |
| flagship | arenahard_math/287 | arenahard_math | intern-s1 | 164 | 39 | 4.20513 |
| flagship | arenahard_creative_writing/523 | arenahard_creative_writing | intern-s1 | 162 | 41 | 3.95122 |
| flagship | arenahard_creative_writing/668 | arenahard_creative_writing | intern-s1 | 162 | 43 | 3.76744 |
| flagship | arenahard_math/333 | arenahard_math | intern-s1 | 171 | 48 | 3.5625 |
| flagship | arenahard_math/289 | arenahard_math | intern-s1 | 170 | 48 | 3.54167 |
| flagship | arenahard_math/311 | arenahard_math | intern-s1 | 177 | 50 | 3.54 |
| flagship | arenahard_math/365 | arenahard_math | intern-s1 | 186 | 54 | 3.44444 |
| flagship | arenahard_creative_writing/697 | arenahard_creative_writing | intern-s1 | 164 | 48 | 3.41667 |
| flagship | arenahard_math/420 | arenahard_math | intern-s1 | 170 | 50 | 3.4 |
| flagship | arenahard_creative_writing/750 | arenahard_creative_writing | intern-s1 | 163 | 49 | 3.32653 |
| flagship | arenahard_math/454 | arenahard_math | intern-s1 | 167 | 51 | 3.27451 |
| flagship | simpleqa/3757 | simpleqa | intern-s1 | 162 | 50 | 3.24 |
| flagship | arenahard_creative_writing/592 | arenahard_creative_writing | intern-s1 | 163 | 51 | 3.19608 |
| flagship | arenahard_creative_writing/674 | arenahard_creative_writing | intern-s1 | 163 | 51 | 3.19608 |
| flagship | arenahard_math/394 | arenahard_math | intern-s1 | 181 | 57 | 3.17544 |
| flagship | arenahard_math/431 | arenahard_math | intern-s1 | 173 | 56 | 3.08929 |
| flagship | simpleqa/1112 | simpleqa | intern-s1 | 169 | 55 | 3.07273 |
| flagship | arenahard_creative_writing/599 | arenahard_creative_writing | intern-s1 | 168 | 55 | 3.05455 |
| flagship | arenahard_creative_writing/625 | arenahard_creative_writing | intern-s1 | 164 | 54 | 3.03704 |
| mixed | livemathbench/7 | livemathbench | gpt-5 | 3295 | 647 | 5.09274 |
| mixed | arenahard_math/287 | arenahard_math | intern-s1 | 164 | 39 | 4.20513 |
| mixed | arenahard_creative_writing/523 | arenahard_creative_writing | intern-s1 | 162 | 41 | 3.95122 |
| mixed | arenahard_creative_writing/668 | arenahard_creative_writing | intern-s1 | 162 | 43 | 3.76744 |
| mixed | arenahard_math/333 | arenahard_math | intern-s1 | 171 | 48 | 3.5625 |
| mixed | arenahard_math/289 | arenahard_math | intern-s1 | 170 | 48 | 3.54167 |
| mixed | arenahard_math/311 | arenahard_math | intern-s1 | 177 | 50 | 3.54 |
| mixed | arenahard_math/365 | arenahard_math | intern-s1 | 186 | 54 | 3.44444 |
| mixed | arenahard_creative_writing/697 | arenahard_creative_writing | intern-s1 | 164 | 48 | 3.41667 |
| mixed | arenahard_math/420 | arenahard_math | intern-s1 | 170 | 50 | 3.4 |
| mixed | arenahard_creative_writing/750 | arenahard_creative_writing | intern-s1 | 163 | 49 | 3.32653 |
| mixed | arenahard_math/454 | arenahard_math | intern-s1 | 167 | 51 | 3.27451 |
| mixed | arenahard_creative_writing/592 | arenahard_creative_writing | intern-s1 | 163 | 51 | 3.19608 |
| mixed | arenahard_creative_writing/674 | arenahard_creative_writing | intern-s1 | 163 | 51 | 3.19608 |
| mixed | arenahard_math/394 | arenahard_math | intern-s1 | 181 | 57 | 3.17544 |
| mixed | arenahard_math/431 | arenahard_math | intern-s1 | 173 | 56 | 3.08929 |
| mixed | arenahard_creative_writing/599 | arenahard_creative_writing | intern-s1 | 168 | 55 | 3.05455 |
| mixed | arenahard_creative_writing/625 | arenahard_creative_writing | intern-s1 | 164 | 54 | 3.03704 |
| mixed | arenahard_creative_writing/501 | arenahard_creative_writing | intern-s1 | 166 | 56 | 2.96429 |
| mixed | arenahard_math/417 | arenahard_math | intern-s1 | 182 | 62 | 2.93548 |

## Prompt overhead per model

| pool | model | prompt_overhead_tokens | refit_without_exclusion |
| --- | --- | --- | --- |
| flagship | claude-sonnet-4 | 0 | 0 |
| flagship | deepseek-r1-0528 | 0 | 0 |
| flagship | deepseek-v3-0324 | 0 | 0 |
| flagship | gemini-2.5-flash | 0 | 0 |
| flagship | gemini-2.5-pro | 0 | 0 |
| flagship | glm-4.6 | 0 | 0 |
| flagship | gpt-5 | 0 | 0 |
| flagship | gpt-5-chat | 0 | 0 |
| flagship | intern-s1 | 124 | 124 |
| flagship | kimi-k2-0905 | 0 | 0 |
| flagship | qwen3-235b-a22b-2507 | 0 | 0 |
| flagship | qwen3-235b-a22b-thinking-2507 | 0 | 0 |
| mixed | DeepSeek-R1-0528-Qwen3-8B | 0 | 0 |
| mixed | Llama-3.1-Nemotron-Nano-8B-v1 | 0 | 0 |
| mixed | NVIDIA-Nemotron-Nano-9B-v2 | 0 | 0 |
| mixed | Qwen3-8B | 0 | 0 |
| mixed | claude-sonnet-4 | 0 | 0 |
| mixed | deepseek-r1-0528 | 0 | 0 |
| mixed | deepseek-v3-0324 | 0 | 0 |
| mixed | gemini-2.5-flash | 0 | 0 |
| mixed | gemini-2.5-pro | 0 | 0 |
| mixed | glm-4.6 | 0 | 0 |
| mixed | gpt-5 | 0 | 0 |
| mixed | gpt-5-chat | 0 | 0 |
| mixed | intern-s1 | 124 | 124 |
| mixed | kimi-k2-0905 | 0 | 0 |
| mixed | qwen3-235b-a22b-2507 | 0 | 0 |
| mixed | qwen3-235b-a22b-thinking-2507 | 0 | 0 |

## Adjusted ratio per split

| pool | split | records | max_ratio | p50 | p99 | p99.9 | count_above_1 | coverage | anomalies | count_above_1_without_anomalies | coverage_without_anomalies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | train | 61268 | 1 | 0.26375 | 0.89854 | 0.958745 | 0 | 1 | 0 | 0 | 1 |
| flagship | calib | 20952 | 0.988372 | 0.264648 | 0.888889 | 0.944648 | 0 | 1 | 0 | 0 | 1 |
| flagship | test | 20518 | 1.04494 | 0.263457 | 0.911561 | 0.958014 | 3 | 0.999854 | 0 | 3 | 0.999854 |
| flagship | heldout | 40188 | 5.09274 | 0.282144 | 0.579414 | 0.674068 | 1 | 0.999975 | 1 | 0 | 1 |
| mixed | train | 16904 | 1 | 0.284908 | 0.602324 | 0.868516 | 0 | 1 | 0 | 0 | 1 |
| mixed | calib | 6050 | 0.988372 | 0.28932 | 0.601255 | 0.907355 | 0 | 1 | 0 | 0 | 1 |
| mixed | test | 5724 | 1.04494 | 0.286832 | 0.631579 | 0.92539 | 3 | 0.999476 | 0 | 3 | 0.999476 |
| mixed | heldout | 21843 | 5.09274 | 0.271795 | 0.573546 | 0.733654 | 1 | 0.999954 | 1 | 0 | 1 |

## Sensitivity: overhead fitted without excluding any record

| pool | split | records | max_ratio | p50 | p99 | p99.9 | count_above_1 | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | train | 61268 | 1 | 0.26375 | 0.89854 | 0.958745 | 0 | 1 |
| flagship | calib | 20952 | 0.988372 | 0.264648 | 0.888889 | 0.944648 | 0 | 1 |
| flagship | test | 20518 | 1.04494 | 0.263457 | 0.911561 | 0.958014 | 3 | 0.999854 |
| flagship | heldout | 40188 | 5.09274 | 0.282144 | 0.579414 | 0.674068 | 1 | 0.999975 |
| mixed | train | 16904 | 1 | 0.284908 | 0.602324 | 0.868516 | 0 | 1 |
| mixed | calib | 6050 | 0.988372 | 0.28932 | 0.601255 | 0.907355 | 0 | 1 |
| mixed | test | 5724 | 1.04494 | 0.286832 | 0.631579 | 0.92539 | 3 | 0.999476 |
| mixed | heldout | 21843 | 5.09274 | 0.271795 | 0.573546 | 0.733654 | 1 | 0.999954 |

## Records above 1 after the overhead (worst first)

| pool | item_id | dataset | split | model | prompt_tokens | in_upper | prompt_overhead | bound | ratio | anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | livemathbench/7 | livemathbench | heldout | gpt-5 | 3295 | 647 | 0 | 647 | 5.09274 | True |
| flagship | arenahard_math/365 | arenahard_math | test | intern-s1 | 186 | 54 | 124 | 178 | 1.04494 | False |
| flagship | arenahard_math/311 | arenahard_math | test | intern-s1 | 177 | 50 | 124 | 174 | 1.01724 | False |
| flagship | arenahard_math/287 | arenahard_math | test | intern-s1 | 164 | 39 | 124 | 163 | 1.00613 | False |
| mixed | livemathbench/7 | livemathbench | heldout | gpt-5 | 3295 | 647 | 0 | 647 | 5.09274 | True |
| mixed | arenahard_math/365 | arenahard_math | test | intern-s1 | 186 | 54 | 124 | 178 | 1.04494 | False |
| mixed | arenahard_math/311 | arenahard_math | test | intern-s1 | 177 | 50 | 124 | 174 | 1.01724 | False |
| mixed | arenahard_math/287 | arenahard_math | test | intern-s1 | 164 | 39 | 124 | 163 | 1.00613 | False |

## Anomalous records

| pool | split | model | anomalies |
| --- | --- | --- | --- |
| flagship | heldout | gpt-5 | 1 |
| mixed | heldout | gpt-5 | 1 |

## Largest anomalous records

| pool | split | item_id | dataset | model | prompt_tokens | others_median |
| --- | --- | --- | --- | --- | --- | --- |
| flagship | heldout | livemathbench/7 | livemathbench | gpt-5 | 3295 | 180.5 |
| mixed | heldout | livemathbench/7 | livemathbench | gpt-5 | 3295 | 182 |
