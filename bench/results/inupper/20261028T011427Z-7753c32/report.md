# inupper

Run `20261028T011427Z-7753c32`.

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
| flagship | intern-s1 | 68 | 124 |
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
| mixed | intern-s1 | 68 | 124 |
| mixed | kimi-k2-0905 | 0 | 0 |
| mixed | qwen3-235b-a22b-2507 | 0 | 0 |
| mixed | qwen3-235b-a22b-thinking-2507 | 0 | 0 |

## Adjusted ratio per split

| pool | split | records | max_ratio | p50 | p99 | p99.9 | count_above_1 | coverage | anomalies | count_above_1_without_anomalies | coverage_without_anomalies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flagship | train | 61268 | 1.47414 | 0.263804 | 1.09109 | 1.26614 | 1248 | 0.97963 | 2779 | 0 | 1 |
| flagship | calib | 20952 | 1.48624 | 0.264648 | 1.10256 | 1.2592 | 450 | 0.978522 | 941 | 0 | 1 |
| flagship | test | 20518 | 1.53271 | 0.263457 | 1.09554 | 1.28725 | 430 | 0.979043 | 937 | 0 | 1 |
| flagship | heldout | 40188 | 5.09274 | 0.282723 | 0.63404 | 0.808896 | 14 | 0.999652 | 84 | 0 | 1 |
| mixed | train | 16904 | 1.47414 | 0.284959 | 0.640572 | 1.18893 | 41 | 0.997575 | 143 | 0 | 1 |
| mixed | calib | 6050 | 1.48624 | 0.289407 | 0.689028 | 1.24976 | 19 | 0.99686 | 55 | 0 | 1 |
| mixed | test | 5724 | 1.53271 | 0.286832 | 0.690351 | 1.312 | 17 | 0.99703 | 58 | 0 | 1 |
| mixed | heldout | 21843 | 5.09274 | 0.272109 | 0.621142 | 0.911812 | 14 | 0.999359 | 82 | 0 | 1 |

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
| flagship | arenahard_math/287 | arenahard_math | test | intern-s1 | 164 | 39 | 68 | 107 | 1.53271 | True |
| flagship | arenahard_math/365 | arenahard_math | test | intern-s1 | 186 | 54 | 68 | 122 | 1.52459 | True |
| flagship | arenahard_math/311 | arenahard_math | test | intern-s1 | 177 | 50 | 68 | 118 | 1.5 | True |
| flagship | arenahard_creative_writing/523 | arenahard_creative_writing | calib | intern-s1 | 162 | 41 | 68 | 109 | 1.48624 | True |
| flagship | arenahard_math/333 | arenahard_math | train | intern-s1 | 171 | 48 | 68 | 116 | 1.47414 | True |
| flagship | arenahard_math/289 | arenahard_math | calib | intern-s1 | 170 | 48 | 68 | 116 | 1.46552 | True |
| flagship | arenahard_creative_writing/668 | arenahard_creative_writing | calib | intern-s1 | 162 | 43 | 68 | 111 | 1.45946 | True |
| flagship | arenahard_math/394 | arenahard_math | train | intern-s1 | 181 | 57 | 68 | 125 | 1.448 | True |
| flagship | arenahard_math/420 | arenahard_math | train | intern-s1 | 170 | 50 | 68 | 118 | 1.44068 | True |
| flagship | arenahard_creative_writing/697 | arenahard_creative_writing | train | intern-s1 | 164 | 48 | 68 | 116 | 1.41379 | True |
| flagship | arenahard_math/454 | arenahard_math | calib | intern-s1 | 167 | 51 | 68 | 119 | 1.40336 | True |
| flagship | arenahard_math/417 | arenahard_math | test | intern-s1 | 182 | 62 | 68 | 130 | 1.4 | True |
| flagship | arenahard_math/431 | arenahard_math | train | intern-s1 | 173 | 56 | 68 | 124 | 1.39516 | True |
| flagship | arenahard_creative_writing/750 | arenahard_creative_writing | calib | intern-s1 | 163 | 49 | 68 | 117 | 1.39316 | True |
| flagship | simpleqa/1112 | simpleqa | train | intern-s1 | 169 | 55 | 68 | 123 | 1.37398 | True |
| flagship | simpleqa/3757 | simpleqa | train | intern-s1 | 162 | 50 | 68 | 118 | 1.37288 | True |
| flagship | arenahard_creative_writing/592 | arenahard_creative_writing | test | intern-s1 | 163 | 51 | 68 | 119 | 1.36975 | True |
| flagship | arenahard_creative_writing/674 | arenahard_creative_writing | train | intern-s1 | 163 | 51 | 68 | 119 | 1.36975 | True |
| flagship | arenahard_creative_writing/599 | arenahard_creative_writing | train | intern-s1 | 168 | 55 | 68 | 123 | 1.36585 | True |
| mixed | livemathbench/7 | livemathbench | heldout | gpt-5 | 3295 | 647 | 0 | 647 | 5.09274 | True |
| mixed | arenahard_math/287 | arenahard_math | test | intern-s1 | 164 | 39 | 68 | 107 | 1.53271 | True |
| mixed | arenahard_math/365 | arenahard_math | test | intern-s1 | 186 | 54 | 68 | 122 | 1.52459 | True |
| mixed | arenahard_math/311 | arenahard_math | test | intern-s1 | 177 | 50 | 68 | 118 | 1.5 | True |
| mixed | arenahard_creative_writing/523 | arenahard_creative_writing | calib | intern-s1 | 162 | 41 | 68 | 109 | 1.48624 | True |
| mixed | arenahard_math/333 | arenahard_math | train | intern-s1 | 171 | 48 | 68 | 116 | 1.47414 | True |
| mixed | arenahard_math/289 | arenahard_math | calib | intern-s1 | 170 | 48 | 68 | 116 | 1.46552 | True |
| mixed | arenahard_creative_writing/668 | arenahard_creative_writing | calib | intern-s1 | 162 | 43 | 68 | 111 | 1.45946 | True |
| mixed | arenahard_math/394 | arenahard_math | train | intern-s1 | 181 | 57 | 68 | 125 | 1.448 | True |
| mixed | arenahard_math/420 | arenahard_math | train | intern-s1 | 170 | 50 | 68 | 118 | 1.44068 | True |
| mixed | arenahard_creative_writing/697 | arenahard_creative_writing | train | intern-s1 | 164 | 48 | 68 | 116 | 1.41379 | True |
| mixed | arenahard_math/454 | arenahard_math | calib | intern-s1 | 167 | 51 | 68 | 119 | 1.40336 | True |
| mixed | arenahard_math/417 | arenahard_math | test | intern-s1 | 182 | 62 | 68 | 130 | 1.4 | True |
| mixed | arenahard_math/431 | arenahard_math | train | intern-s1 | 173 | 56 | 68 | 124 | 1.39516 | True |
| mixed | arenahard_creative_writing/750 | arenahard_creative_writing | calib | intern-s1 | 163 | 49 | 68 | 117 | 1.39316 | True |
| mixed | arenahard_creative_writing/592 | arenahard_creative_writing | test | intern-s1 | 163 | 51 | 68 | 119 | 1.36975 | True |
| mixed | arenahard_creative_writing/674 | arenahard_creative_writing | train | intern-s1 | 163 | 51 | 68 | 119 | 1.36975 | True |
| mixed | arenahard_creative_writing/599 | arenahard_creative_writing | train | intern-s1 | 168 | 55 | 68 | 123 | 1.36585 | True |
| mixed | arenahard_creative_writing/625 | arenahard_creative_writing | train | intern-s1 | 164 | 54 | 68 | 122 | 1.34426 | True |
| mixed | arenahard_creative_writing/501 | arenahard_creative_writing | train | intern-s1 | 166 | 56 | 68 | 124 | 1.33871 | True |

## Anomalous records

| pool | split | model | anomalies |
| --- | --- | --- | --- |
| flagship | train | intern-s1 | 2777 |
| flagship | train | kimi-k2-0905 | 2 |
| flagship | calib | intern-s1 | 939 |
| flagship | calib | kimi-k2-0905 | 2 |
| flagship | test | intern-s1 | 937 |
| flagship | heldout | gpt-5 | 1 |
| flagship | heldout | intern-s1 | 83 |
| mixed | train | intern-s1 | 143 |
| mixed | calib | intern-s1 | 55 |
| mixed | test | intern-s1 | 58 |
| mixed | heldout | gpt-5 | 1 |
| mixed | heldout | intern-s1 | 81 |

## Largest anomalous records

| pool | split | item_id | dataset | model | prompt_tokens | others_median |
| --- | --- | --- | --- | --- | --- | --- |
| flagship | train | simpleqa/3757 | simpleqa | intern-s1 | 162 | 11 |
| flagship | train | arenahard_creative_writing/599 | arenahard_creative_writing | intern-s1 | 168 | 12 |
| flagship | train | simpleqa/101 | simpleqa | intern-s1 | 164 | 13 |
| flagship | train | arenahard_creative_writing/697 | arenahard_creative_writing | intern-s1 | 164 | 13.5 |
| flagship | train | simpleqa/1897 | simpleqa | intern-s1 | 166 | 14 |
| flagship | train | simpleqa/1279 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/1492 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/1501 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/1588 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/2072 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/3767 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/3785 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/3872 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | simpleqa/4154 | simpleqa | intern-s1 | 165 | 14 |
| flagship | train | arenahard_creative_writing/625 | arenahard_creative_writing | intern-s1 | 164 | 14 |
| flagship | train | arenahard_creative_writing/674 | arenahard_creative_writing | intern-s1 | 163 | 14 |
| flagship | train | arenahard_creative_writing/646 | arenahard_creative_writing | intern-s1 | 168 | 15 |
| flagship | train | simpleqa/1324 | simpleqa | intern-s1 | 168 | 15 |
| flagship | train | simpleqa/2688 | simpleqa | intern-s1 | 167 | 15 |
| flagship | train | arenahard_creative_writing/501 | arenahard_creative_writing | intern-s1 | 166 | 15 |
| flagship | calib | simpleqa/4277 | simpleqa | intern-s1 | 163 | 12 |
| flagship | calib | arenahard_creative_writing/523 | arenahard_creative_writing | intern-s1 | 162 | 12 |
| flagship | calib | arenahard_creative_writing/668 | arenahard_creative_writing | intern-s1 | 162 | 12 |
| flagship | calib | simpleqa/1568 | simpleqa | intern-s1 | 164 | 13 |
| flagship | calib | arenahard_creative_writing/750 | arenahard_creative_writing | intern-s1 | 163 | 13 |
| flagship | calib | simpleqa/1281 | simpleqa | intern-s1 | 165 | 14 |
| flagship | calib | simpleqa/1748 | simpleqa | intern-s1 | 165 | 14 |
| flagship | calib | simpleqa/2889 | simpleqa | intern-s1 | 165 | 14 |
| flagship | calib | simpleqa/3857 | simpleqa | intern-s1 | 165 | 14 |
| flagship | calib | simpleqa/4115 | simpleqa | intern-s1 | 165 | 14 |
| flagship | calib | simpleqa/1043 | simpleqa | intern-s1 | 166 | 15 |
| flagship | calib | simpleqa/185 | simpleqa | intern-s1 | 166 | 15 |
| flagship | calib | simpleqa/2958 | simpleqa | intern-s1 | 166 | 15 |
| flagship | calib | simpleqa/3028 | simpleqa | intern-s1 | 166 | 15 |
| flagship | calib | simpleqa/2337 | simpleqa | intern-s1 | 168 | 16 |
| flagship | calib | simpleqa/968 | simpleqa | intern-s1 | 168 | 16 |
| flagship | calib | arenahard_math/353 | arenahard_math | intern-s1 | 167 | 16 |
| flagship | calib | arenahard_math/454 | arenahard_math | intern-s1 | 167 | 16 |
| flagship | calib | simpleqa/1142 | simpleqa | intern-s1 | 167 | 16 |
| flagship | calib | simpleqa/1146 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | arenahard_creative_writing/592 | arenahard_creative_writing | intern-s1 | 163 | 13 |
| flagship | test | simpleqa/2107 | simpleqa | intern-s1 | 166 | 14 |
| flagship | test | simpleqa/3817 | simpleqa | intern-s1 | 166 | 14 |
| flagship | test | simpleqa/56 | simpleqa | intern-s1 | 166 | 14 |
| flagship | test | arenahard_creative_writing/634 | arenahard_creative_writing | intern-s1 | 164 | 14 |
| flagship | test | arenahard_creative_writing/654 | arenahard_creative_writing | intern-s1 | 164 | 14 |
| flagship | test | arenahard_math/287 | arenahard_math | intern-s1 | 164 | 14 |
| flagship | test | simpleqa/4026 | simpleqa | intern-s1 | 168 | 15 |
| flagship | test | simpleqa/864 | simpleqa | intern-s1 | 169 | 16 |
| flagship | test | simpleqa/1062 | simpleqa | intern-s1 | 168 | 16 |
| flagship | test | simpleqa/1156 | simpleqa | intern-s1 | 168 | 16 |
| flagship | test | simpleqa/4054 | simpleqa | intern-s1 | 168 | 16 |
| flagship | test | simpleqa/1212 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/1366 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/2846 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/3200 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/3247 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/3530 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/4113 | simpleqa | intern-s1 | 167 | 16 |
| flagship | test | simpleqa/4182 | simpleqa | intern-s1 | 167 | 16 |
| flagship | heldout | livemathbench/7 | livemathbench | gpt-5 | 3295 | 180.5 |
| flagship | heldout | arenahard_coding/107 | arenahard_coding | intern-s1 | 166 | 16 |
| flagship | heldout | arenahard_coding/186 | arenahard_coding | intern-s1 | 166 | 16 |
| flagship | heldout | arenahard_coding/116 | arenahard_coding | intern-s1 | 169 | 19 |
| flagship | heldout | arenahard_coding/137 | arenahard_coding | intern-s1 | 169 | 19 |
| flagship | heldout | arenahard_coding/64 | arenahard_coding | intern-s1 | 169 | 19 |
| flagship | heldout | arenahard_coding/104 | arenahard_coding | intern-s1 | 170 | 20 |
| flagship | heldout | arenahard_coding/246 | arenahard_coding | intern-s1 | 170 | 20 |
| flagship | heldout | arenahard_coding/15 | arenahard_coding | intern-s1 | 173 | 21 |
| flagship | heldout | arenahard_coding/26 | arenahard_coding | intern-s1 | 174 | 22 |
| flagship | heldout | arenahard_coding/215 | arenahard_coding | intern-s1 | 173 | 22 |
| flagship | heldout | arenahard_coding/63 | arenahard_coding | intern-s1 | 173 | 22 |
| flagship | heldout | arenahard_coding/78 | arenahard_coding | intern-s1 | 172 | 22 |
| flagship | heldout | arenahard_coding/56 | arenahard_coding | intern-s1 | 176 | 25 |
| flagship | heldout | arenahard_coding/135 | arenahard_coding | intern-s1 | 176 | 26 |
| flagship | heldout | arenahard_coding/183 | arenahard_coding | intern-s1 | 176 | 26 |
| flagship | heldout | arenahard_coding/187 | arenahard_coding | intern-s1 | 176 | 26 |
| flagship | heldout | arenahard_coding/82 | arenahard_coding | intern-s1 | 176 | 26 |
| flagship | heldout | arenahard_coding/2 | arenahard_coding | intern-s1 | 178 | 27 |
| flagship | heldout | arenahard_coding/125 | arenahard_coding | intern-s1 | 177 | 27 |
| mixed | train | arenahard_creative_writing/674 | arenahard_creative_writing | intern-s1 | 163 | 14 |
| mixed | train | arenahard_creative_writing/697 | arenahard_creative_writing | intern-s1 | 164 | 14.5 |
| mixed | train | arenahard_creative_writing/625 | arenahard_creative_writing | intern-s1 | 164 | 15 |
| mixed | train | arenahard_creative_writing/599 | arenahard_creative_writing | intern-s1 | 168 | 16 |
| mixed | train | arenahard_creative_writing/646 | arenahard_creative_writing | intern-s1 | 168 | 16 |
| mixed | train | arenahard_creative_writing/501 | arenahard_creative_writing | intern-s1 | 166 | 16 |
| mixed | train | arenahard_creative_writing/529 | arenahard_creative_writing | intern-s1 | 165 | 16 |
| mixed | train | arenahard_math/442 | arenahard_math | intern-s1 | 168 | 17.5 |
| mixed | train | arenahard_math/351 | arenahard_math | intern-s1 | 169 | 19.5 |
| mixed | train | arenahard_creative_writing/721 | arenahard_creative_writing | intern-s1 | 170 | 20 |
| mixed | train | arenahard_creative_writing/597 | arenahard_creative_writing | intern-s1 | 169 | 20 |
| mixed | train | arenahard_creative_writing/618 | arenahard_creative_writing | intern-s1 | 169 | 20 |
| mixed | train | arenahard_math/394 | arenahard_math | intern-s1 | 181 | 22 |
| mixed | train | arenahard_math/333 | arenahard_math | intern-s1 | 171 | 21 |
| mixed | train | arenahard_creative_writing/550 | arenahard_creative_writing | intern-s1 | 170 | 21 |
| mixed | train | arenahard_creative_writing/718 | arenahard_creative_writing | intern-s1 | 170 | 21 |
| mixed | train | arenahard_creative_writing/734 | arenahard_creative_writing | intern-s1 | 170 | 21 |
| mixed | train | arenahard_math/453 | arenahard_math | intern-s1 | 173 | 22 |
| mixed | train | arenahard_math/309 | arenahard_math | intern-s1 | 172 | 22 |
| mixed | train | arenahard_creative_writing/544 | arenahard_creative_writing | intern-s1 | 171 | 22 |
| mixed | calib | arenahard_creative_writing/750 | arenahard_creative_writing | intern-s1 | 163 | 13 |
| mixed | calib | arenahard_creative_writing/523 | arenahard_creative_writing | intern-s1 | 162 | 13 |
| mixed | calib | arenahard_creative_writing/668 | arenahard_creative_writing | intern-s1 | 162 | 13 |
| mixed | calib | arenahard_creative_writing/732 | arenahard_creative_writing | intern-s1 | 167 | 17 |
| mixed | calib | arenahard_math/353 | arenahard_math | intern-s1 | 167 | 17 |
| mixed | calib | arenahard_math/445 | arenahard_math | intern-s1 | 165 | 17 |
| mixed | calib | arenahard_math/270 | arenahard_math | intern-s1 | 167 | 18 |
| mixed | calib | arenahard_math/454 | arenahard_math | intern-s1 | 167 | 18 |
| mixed | calib | arenahard_math/289 | arenahard_math | intern-s1 | 170 | 19 |
| mixed | calib | arenahard_creative_writing/627 | arenahard_creative_writing | intern-s1 | 170 | 20 |
| mixed | calib | arenahard_creative_writing/516 | arenahard_creative_writing | intern-s1 | 172 | 23 |
| mixed | calib | arenahard_creative_writing/706 | arenahard_creative_writing | intern-s1 | 173 | 24 |
| mixed | calib | arenahard_math/324 | arenahard_math | intern-s1 | 174 | 25 |
| mixed | calib | arenahard_creative_writing/730 | arenahard_creative_writing | intern-s1 | 175 | 26 |
| mixed | calib | arenahard_math/272 | arenahard_math | intern-s1 | 175 | 26 |
| mixed | calib | arenahard_math/444 | arenahard_math | intern-s1 | 176 | 26.5 |
| mixed | calib | arenahard_creative_writing/666 | arenahard_creative_writing | intern-s1 | 176 | 27 |
| mixed | calib | arenahard_math/392 | arenahard_math | intern-s1 | 179 | 31 |
| mixed | calib | arenahard_creative_writing/715 | arenahard_creative_writing | intern-s1 | 187 | 33 |
| mixed | calib | arenahard_math/317 | arenahard_math | intern-s1 | 185 | 33 |
| mixed | test | arenahard_creative_writing/592 | arenahard_creative_writing | intern-s1 | 163 | 13 |
| mixed | test | arenahard_creative_writing/634 | arenahard_creative_writing | intern-s1 | 164 | 14 |
| mixed | test | arenahard_creative_writing/654 | arenahard_creative_writing | intern-s1 | 164 | 15 |
| mixed | test | arenahard_math/287 | arenahard_math | intern-s1 | 164 | 15 |
| mixed | test | arenahard_creative_writing/748 | arenahard_creative_writing | intern-s1 | 172 | 18 |
| mixed | test | arenahard_math/311 | arenahard_math | intern-s1 | 177 | 19.5 |
| mixed | test | arenahard_math/397 | arenahard_math | intern-s1 | 170 | 21 |
| mixed | test | arenahard_math/319 | arenahard_math | intern-s1 | 172 | 22 |
| mixed | test | arenahard_creative_writing/739 | arenahard_creative_writing | intern-s1 | 173 | 23 |
| mixed | test | arenahard_creative_writing/703 | arenahard_creative_writing | intern-s1 | 172 | 23 |
| mixed | test | arenahard_math/332 | arenahard_math | intern-s1 | 172 | 23 |
| mixed | test | arenahard_math/337 | arenahard_math | intern-s1 | 174 | 24 |
| mixed | test | arenahard_creative_writing/682 | arenahard_creative_writing | intern-s1 | 178 | 26 |
| mixed | test | arenahard_creative_writing/562 | arenahard_creative_writing | intern-s1 | 177 | 27 |
| mixed | test | arenahard_creative_writing/553 | arenahard_creative_writing | intern-s1 | 177 | 28 |
| mixed | test | arenahard_math/455 | arenahard_math | intern-s1 | 181 | 30 |
| mixed | test | arenahard_math/352 | arenahard_math | intern-s1 | 181 | 31 |
| mixed | test | arenahard_math/365 | arenahard_math | intern-s1 | 186 | 32 |
| mixed | test | arenahard_math/417 | arenahard_math | intern-s1 | 182 | 32 |
| mixed | test | arenahard_creative_writing/540 | arenahard_creative_writing | intern-s1 | 187 | 34 |
| mixed | heldout | livemathbench/7 | livemathbench | gpt-5 | 3295 | 182 |
| mixed | heldout | arenahard_coding/186 | arenahard_coding | intern-s1 | 166 | 16 |
| mixed | heldout | arenahard_coding/107 | arenahard_coding | intern-s1 | 166 | 17 |
| mixed | heldout | arenahard_coding/64 | arenahard_coding | intern-s1 | 169 | 19 |
| mixed | heldout | arenahard_coding/104 | arenahard_coding | intern-s1 | 170 | 20 |
| mixed | heldout | arenahard_coding/246 | arenahard_coding | intern-s1 | 170 | 20 |
| mixed | heldout | arenahard_coding/116 | arenahard_coding | intern-s1 | 169 | 20 |
| mixed | heldout | arenahard_coding/137 | arenahard_coding | intern-s1 | 169 | 20 |
| mixed | heldout | arenahard_coding/15 | arenahard_coding | intern-s1 | 173 | 21 |
| mixed | heldout | arenahard_coding/78 | arenahard_coding | intern-s1 | 172 | 22 |
| mixed | heldout | arenahard_coding/215 | arenahard_coding | intern-s1 | 173 | 23 |
| mixed | heldout | arenahard_coding/63 | arenahard_coding | intern-s1 | 173 | 24 |
| mixed | heldout | arenahard_coding/26 | arenahard_coding | intern-s1 | 174 | 25 |
| mixed | heldout | arenahard_coding/183 | arenahard_coding | intern-s1 | 176 | 26 |
| mixed | heldout | arenahard_coding/195 | arenahard_coding | intern-s1 | 177 | 27 |
| mixed | heldout | arenahard_coding/241 | arenahard_coding | intern-s1 | 177 | 27 |
| mixed | heldout | arenahard_coding/135 | arenahard_coding | intern-s1 | 176 | 27 |
| mixed | heldout | arenahard_coding/187 | arenahard_coding | intern-s1 | 176 | 27 |
| mixed | heldout | arenahard_coding/56 | arenahard_coding | intern-s1 | 176 | 27 |
| mixed | heldout | arenahard_coding/82 | arenahard_coding | intern-s1 | 176 | 27 |
