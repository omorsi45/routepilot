# clock

Run `20261029T074230Z-e12fd9f`.

**Labels:** measured on this host with rp-load --clock-probe, which reads time.Now for 200 ms and reports the smallest non-zero difference between two consecutive reads (the clock step) and the share of equal consecutive reads; nothing here is simulated, fitted or assumed; a step above 1000 ns is too coarse for the sub-millisecond quantities of the performance experiments, which refuse to run on it.

## Probes

| probe | clock_step_ns | zero_fraction |
| --- | --- | --- |
| 1 | 501300 | 0.999993 |
| 2 | 502500 | 0.99999 |
| 3 | 502000 | 0.999991 |
| 4 | 379700 | 0.999992 |
| 5 | 501500 | 0.999992 |

## Summary

| probes | min_step_ns | median_step_ns | max_step_ns | max_zero_fraction | threshold_ns | coarse |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 379700 | 501500 | 502500 | 0.999993 | 1000 | True |

## Host

| os | system | go_version |
| --- | --- | --- |
| Windows-11-10.0.26300-SP0 | Windows | go1.27.2 |
