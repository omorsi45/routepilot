# live_collect

Run `20261029T015819Z-00deb40`.

**Labels:** speeds and times are measured on this machine by sending requests straight to Ollama, not through RoutePilot; the three measurement prompts are invented, not dataset items; no prompt text and no model output is in this report.

## Candidate models, measured directly on Ollama

| model | digest | parameters | TTFT ms (mean of 3) | output tokens/s (mean of 3) | output tokens (mean) | answers right (of 3) | request s at 384 tokens | load s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:0.5b | a8b0c5157701 | 494.03M | 36.8121 | 174.947 | 223 | 2 | 2.23177 | 2.50626 |
| qwen2.5:1.5b | 65ec06548149 | 1.5B | 50.6056 | 97.0543 | 71.3333 | 3 | 4.00715 | 2.6053 |
| qwen2.5:3b | 357c53fb659c | 3.1B | 55.1624 | 77.85 | 153 | 3 | 4.98773 | 3.62996 |
| llama3.2:3b | a80c4f17acd5 | 3.2B | 50.6293 | 73.4404 | 128 | 3 | 5.27936 | 4.17554 |
