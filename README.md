CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Shubham Sharma
  * [LinkedIn](www.linkedin.com/in/codeshubham), [personal website](https://shubhvr.com/).
* Tested on: Windows 10, i7-9750H @ 2.26GHz, 16GB, GTX 1660ti 6GB (Personal Computer).
*GPU Compute Capability: 7.5

## Stream Compaction
This project involves 
-   CPU version of Scan,
-   CPU version of Compact without using the Scan algorithm,
-   CPU version of Compact with Scan,
-   Naive version of Scan,
-   Work-efficient version of Scan, and
-   Work-efficient version of Compact that used the work-efficient Scan's code.

The three cpu calculations are serialized; no multi-threading was consolidated. We have used simple cpu scan and compaction to compare the results with the GPU parallelised algorithm implementation. All the results are then compared. Results of CUDA's Thrust library are also used to compare the execution times of each implementation.   

## Performance Analysis
The projects implements both CPU and GPU timing functions as a performance timer class to conveniently measure the time cost. `std::chrono` is used, to provide CPU high-precision timing and CUDA event to measure the CUDA performance. 
I have collected the data across 8 executions with different array sizes to collect the data. The program generates a new array of random values with each execution, where the size of array is customisable. I have varied the size of the arrays by powers of two, starting from 2^6^ all the wai to 2^28^. The program also executes each algorithm for arrays of size "non- power of two" which are generated truncating the "power of two" arrays. 