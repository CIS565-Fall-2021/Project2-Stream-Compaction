CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Nuofan Xu
* Tested on: Windows 10, AMD Ryzen 3800x @ 3.9Hz 2x16GB RAM, RTX 2080 Super 8GB

#### Overview 

This project implements Scan and Stream Compaction and tests performance of various implementations with different array size and block size( all implementations support Non-Power-Of-Two input). The detailed list is shown below:

*Scan 
    *cpu
    *naive (gpu)
    *work-efficient (gpu, optimized indexing)
    *thrust (gpu)
*Stream Compaction
    *cpu without scan
    *cpu with scan
    *gpu with work-efficient scan

### BlockSize Optimization
A perilimary optimazation is done on the GPU block size parameter. Two different array lengths, 2^8 and 2^18 are used in this step. Through testing, changing blockSize does almost no effect on the performance the performance of Navie and efficient implmentation of scan and stream compaction with small input array size. In the case of big array size, block size does slightly affect the performance. There is no obvious pattern that purely increasing or decreasing block size would lead to a noticeable difference in performance, rather, there seem to be a sweet spot around blozk size 64 to 128. After consideration, bloci size of 128 is used for all the subsequent test results. The graph is plotted as following:
<p float="left">
  <img src="/img/block_size.PNG" width="300" />
  <img src="/img/block_size_big_array.PNG" width="300" /> 
</p>
<!-- ![](img/block_size.PNG) -->

### ArraySize Performance Analysis
Investigations have also been done on array size to see the performance of all implementations. The resulting plot is shown below. The cpu implementation is super fast for small arrays, as it has less memory overheads and data transfers in comparison to the parallelized versions on GPU. When the array size increases, the parallelization begin to manifest its power with complexity O(nlogn) for Navie and O(n) for efficient implemtation. The speed increase caused by parallel processing of array elements on different threads overweighs the cost of memory overhead.

<!-- ![](img/array_length.PNG) -->
<!-- <img src="images/array_length.PNG" width="60%" height="60%"> -->
<p float="left">
  <img src="/img/array_length.PNG" width="300" />
  <img src="/img/array_length_big.PNG" width="300" /> 
</p>

### Work Efficient Scan Optimization
Several optimization attempts have been done to increase the performance on GPU.
* Reduce the number of steps that some threads need to go through.
Not all the threads need to go through the UpSweep and DownSweep part. Threads that are not involved in the process can be terminated early.

* Adjust the blockSize.
Block size in the GPU can be changed to allow a bigger number of threads running in the same block. No obvious effect is obeserved.

* Reduce the number of threads that need to be launched.
This is because not all threads are actually working. For example, if the input size is 1024, we only need 512 threads at most instead of 1024 for the first depth (the number of nodes in the addition tree is only half of the size).

Before those optimaztions, the performance of efficient scan and stream compaction is very low, even lower than the CPU implemention with complexity O(n^2). With the above steps, the performance of parallelized implementations exceeds pure CPU approach at input array size of approximately 2^14 to 2^16. 

### Thrust Libs
Scan and stream compaction is also implemented using thrust library. However, the speed of thrust scan is very slow. The reason behind that, in my opinion, is that these libraries, especially thrust, try to be as generic as possible and optimization often requires specialization: for example a specialization of an algorithm can use shared memory for fundamental types (like int or float) but the generic version can't. Thrust focuses on providing a generic template that can be easily used by all users and sacrifices speed for generalizability.

### Sample Test Result

Sample performance test result of input array size of 2^8 with blockSize of 128
![](img/raw_size_2_8.PNG)

### Feedback
Any feedback on errors in the above analysis or any other places is appreciated.