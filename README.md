CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xiao Wei
* Tested on:  Windows 10, i9-9900k @ 3.6GHz 16.0GB, RTX 2080 SUPER 16GB


Feature
======================
* CPU SCAN and Stream Compaction
* Naive GPU Scan
* Work-Efficient GPU Scan and Stream Compaction
* Thrust scan

Performance Analysis
======================
![PROJECT2](https://user-images.githubusercontent.com/66859615/135018239-b5681125-c5f1-414e-8d9b-87430e9eecd0.jpg)

From the data obtained, we can learn that the rate of change with the growth of array size is slower when we are using GPU methods. The advantage of GPU will probably shows up when the array size grows really huge

From Nsight Profiling, basically it is memory I/O which is the bottleneck. This is better for thrust implementation

![微信图片_20210928114609](https://user-images.githubusercontent.com/66859615/135020062-e14f2ec4-ba5f-4e27-8364-695a14b27ab8.png)


output Example:

****************
** SCAN TESTS **
****************
    [  15  15  16   4  18  27   3  37   8  13  32  30  16 ...  46   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0014ms    (std::chrono Measured)
    [   0  15  30  46  50  68  95  98 135 143 156 188 218 ... 25300 25346 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0016ms    (std::chrono Measured)
    [   0  15  30  46  50  68  95  98 135 143 156 188 218 ... 25262 25280 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.020672ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.018944ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.053248ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.05184ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.044032ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.044896ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   1   0   0   0   3   1   3   0   3   2   2   0 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0022ms    (std::chrono Measured)
    [   3   1   3   1   3   3   2   2   1   1   2   1   1 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0017ms    (std::chrono Measured)
    [   3   1   3   1   3   3   2   2   1   1   2   1   1 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0044ms    (std::chrono Measured)
    [   3   1   3   1   3   3   2   2   1   1   2   1   1 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.058144ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.057856ms    (CUDA Measured)



