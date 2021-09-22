CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jiyu Huang
  * [LinkedIn](https://www.linkedin.com/in/jiyu-huang-0123/)
* Tested on: Windows 10, i7-6700 @ 3.41GHz 16GB, Quadro P1000 4096MB (Towne 062 Lab)

### Overview

This project involves CUDA implementation of GPU parallel algorithms such as scan (prefix sum), stream compaction, and Radix sort. Specifically, the following are implemented:

* CPU scan (serialized, used as comparison)
* CPU stream compact (serialized, used as comparison)
* CPU stream compact with CPU scan (serialized, used as comparison)
* naive version of GPU scan
* work-efficient GPU scan
* work-efficient GPU scan with shared memory and no bank conflict
* GPU stream compaction using optimized work-efficient GPU scan
* GPU Radix sort using optimized work-efficient GPU scan

Thrust library's version of exclusive scan is also used as comparison.

### Performance Analysis

The performance of various implementations of scan are illustrated below.

![performance_chart](/img/performance_chart.png)
![performance_chart_large](/img/performance_chart_large.png)

* As can be seen from the graph, starting from array size 2^15 (32768), GPU algorithms show performance advantages towards the CPU implementation, due to the advantages of parallelism.

* The naive version of GPU scan performs sufficiently well until the array size reaches 2^17 (131072), after which point the performance drops significantly. As shown in the large array graph, it becomes the slowest implementation, even slower than CPU implementation. This is due to the fact that the naive version of GPU scan is not work efficient and in total performs the most amount of computations.

* The work-efficient version of GPU scan initially performs worse than other implementations, but catches up and ends up reducing execution time compared to CPU implemetation and naive GPU implementation. The initial slowness likely results from the extra amount of kernel invocations.

* The shared memory optimization has a significant impact on improving performance and is the fastest implementation in this project, as it should be; operating on shared memory efficiently does prove to be much faster than operating on global memory.

* Thrust library's scan function is almost always the fastest version (except for when the array size is small, or when the array size goes from 2^17 (131072) to 2^18 (262144), where Thrust's scan function has a sudden performance drop).

I would like to delve deeper into the execution timelines for each implementation (and understand why Thrust's scan is so fast), but since I am using the lab computer with no admin access to enable Nsight tracing, I'm temporarily unable to do that.

### Test Results

The following test output is generated with array size of one million (2^20). Extra Radix sort tests (one for power-of-two, one for non-power-of-two) testing the sorting correctness are also included at the end.

```
****************
** SCAN TESTS **
****************
    [  19  18  31  41  16  17   6  12  41   4   7  45  31 ...  11   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.6692ms    (std::chrono Measured)
    [   0  19  37  68 109 125 142 148 160 201 205 212 257 ... 25680674 25680685 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.6274ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 2.83338ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 2.64646ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.11798ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.829952ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.434304ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.463584ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   1   2   1   2   2   1   2   0   3   1   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 3.0586ms    (std::chrono Measured)
    [   2   2   1   2   1   2   2   1   2   3   1   1   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.5556ms    (std::chrono Measured)
    [   2   2   1   2   1   2   2   1   2   3   1   1   1 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 5.8896ms    (std::chrono Measured)
    [   2   2   1   2   1   2   2   1   2   3   1   1   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 1.22845ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 1.25226ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TESTS **
*****************************
    [   2   6  17   6  13  14  18  13   2  16  19   9   1 ...   4   0 ]
==== radix sort, power-of-two ====
    passed
==== radix sort, non-power-of-two ====
    passed
```
