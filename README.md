CUDA Stream Compaction
======================

Implementing GPU stream compaction in CUDA, from scratch. GPU stream compaction is a widely used algorithm, especially for accelerating path tracers.



**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1 - Flocking**

- Anthony Mansur
  - https://www.linkedin.com/in/anthony-mansur-ab3719125/
- Tested on: Windows 10, AMD Ryzen 5 3600, Geforce RTX 2060 Super (personal)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)



### Features 

- CPU implementation of the "prefix sum" algorithm
- CPU implementation of stream compaction with and without use of the scan function
- Naive implementation of the prefix-sum algorithm
- Work-efficient implementation of the prefix-sum algorithm
- GPU implementation of the stream compaction algorithm
- Wrapped the Thrust's scan implementation



### Performance Analysis

Please note: this is an incomplete analysis.

To roughly optimize the block size, compared the the gpu stream compaction algorithm from n = 128 to n = 1024. The following time taken in ms was 5.48, 5.53, 6.86, 5.98, 9.16, 9.20, 8.12, and 7.62. Thus, our block size optimization is of size 128.

Below are the results from running the different algorithms for comparison:

````
****************
** SCAN TESTS **
****************
    [   4  24   5  29  21   6  24  19  39  29  47  46  20 ...   4   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 8.3087ms    (std::chrono Measured)
    [   0   4  28  33  62  83  89 113 132 171 200 247 293 ... 102687260 102687264 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 8.2725ms    (std::chrono Measured)
    [   0   4  28  33  62  83  89 113 132 171 200 247 293 ... 102687181 102687208 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 6.04371ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 6.21773ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 5.66464ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 5.58922ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.254624ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.25872ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   3   3   3   0   0   3   1   1   1   0   0 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 8.4867ms    (std::chrono Measured)
    [   2   3   3   3   3   1   1   1   1   3   1   1   2 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 8.4656ms    (std::chrono Measured)
    [   2   3   3   3   3   1   1   1   1   3   1   1   2 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 8.4656ms    (std::chrono Measured)
    [   2   3   3   3   3   1   1   1   1   3   1   1   2 ...   1   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 5.92832ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 5.76102ms    (CUDA Measured)
    passed
````



### Questions

1. Block Optimization: see performance analysis
2. Comparison of implementations: see performance analysis. Ran with n = 2^22
3. Although there were improvements in performance between naive and work-efficient implementations of scanning, the cpu implementation was faster. This is most likely due to the inefficiencies in terms of branching and in terms of using global memory as opposed to shared memory (i.e., kernels need to be optimized). For compaction, it seems that the gpu implementations ran faster due to the large size of n.
4. See performance analysis for test program output

