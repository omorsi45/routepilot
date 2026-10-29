# clock

Run `20261029T074240Z-e12fd9f`.

**Labels:** measured on this host with rp-load --clock-probe, which reads time.Now for 200 ms and reports the smallest non-zero difference between two consecutive reads (the clock step) and the share of equal consecutive reads; nothing here is simulated, fitted or assumed; a step above 1000 ns is too coarse for the sub-millisecond quantities of the performance experiments, which refuse to run on it.

## Probes

| probe | clock_step_ns | zero_fraction |
| --- | --- | --- |
| 1 | 30 | 0 |
| 2 | 30 | 0 |
| 3 | 30 | 0 |
| 4 | 30 | 0 |
| 5 | 30 | 0 |

## Summary

| probes | min_step_ns | median_step_ns | max_step_ns | max_zero_fraction | threshold_ns | coarse |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 30 | 30 | 30 | 0 | 1000 | False |

## Host

| os | system | go_version |
| --- | --- | --- |
| Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 | Linux | go1.27.2 |
