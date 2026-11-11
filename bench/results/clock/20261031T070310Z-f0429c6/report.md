# clock

Run `20261031T070310Z-f0429c6`.

**Labels:** measured on this host with rp-load --clock-probe, which reads time.Now for 200 ms and reports the smallest non-zero difference between two consecutive reads (the clock step) and the share of equal consecutive reads; nothing here is simulated, fitted or assumed; a step above 1000 ns is too coarse for the sub-millisecond quantities of the performance experiments, which refuse to run on it.

## Probes

| probe | clock_step_ns | zero_fraction |
| --- | --- | --- |
| 1 | 344800 | 0.999993 |
| 2 | 501200 | 0.999992 |
| 3 | 501300 | 0.999992 |
| 4 | 501100 | 0.999993 |
| 5 | 501400 | 0.999993 |

## Summary

| probes | min_step_ns | median_step_ns | max_step_ns | max_zero_fraction | threshold_ns | coarse |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 344800 | 501200 | 501400 | 0.999993 | 1000 | True |

## Host

| os | system | go_version |
| --- | --- | --- |
| Windows-11-10.0.26300-SP0 | Windows | go1.27.2 |
