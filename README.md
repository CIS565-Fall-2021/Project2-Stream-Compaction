CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yuxuan Zhu
  * [LinkedIn](https://www.linkedin.com/in/andrewyxzhu/), [personal website]()
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz 16GB, GTX 1050 4096MB (Personal Laptop)

**Introduction**
This project implemented different versions of the exclusive scanning algorithm and the stream compaction algorithm.
I implemented one CPU version of exclusive scanning and four GPU versions of exclusive scanning.
The CPU version of exclusive scanning supports in-place scanning operation by using local variables. It is surprisingly fast, even for
millions of elements.
The first GPU version of exclusive scanning is naive scanning, which iterativesly sums the array elements and does O(nlog n) computations.
The second GPU version of exclusive scanning is the work efficient version, which consists of an up-sweep portion and a down-sweep portion. I
launched different number of threads for each level of computation to reduce the number of "wasted" threads. 
The third GPU version is done by calling the thrust library. It is highly optimized and extremely fast for large arrays.
The fourth GPU version is optimized based on the second version. I used shared memory to decrease the freqeuncy of global memory access. It is 
roughly twice as fast as the second version. It is also faster than the CPU version for larger arrays. This is considered extra credit.

I also implemented two CPU versions of stream compaction and two GPU versions of stream compaction.
The first CPU version does stream compaction iteratively. It is quite fast.
The second CPU version simulates the stream compaction algorithm on a GPU by using scanning. It is a lot slower.
The first GPU version uses work efficient scanning to implement the stream compaction. It is quite fast.
The second GPU version uses the optimized work efficient scanning to implement stream compaction. It is also faster than the first version.

**Performance Analysis**

I empirically found that the best block size is 256 for all version of scan.

![Scan](img/Scan_Performance.png)

The graph above shows the performance comparsion among the different versions of the algorithm. The CPU version is almost always faster, unless the array
size is extremely large. The thrust library performance is always strictly better than my implementations. This is reasonable since I did not optimize everything due to
my current limited understanding of GPU performance. I don't know what's happening under the hood yet. I tried to analyze performance bottle necks by commenting out certain kernels and checking how much improvement in performance we get. For example, I realized the kernel that adds block increment to each element in a block is very slow. I changed the code to use shared memory to improve the bottleneck due to global memory read/write speed.

![Compaction](img/Stream_Compaction_Performance.png)

The graph above shows the performance comparision among different version of stream compaction. The CPU version was out-performed by my GPU version when the array size is larger than 100000. 

