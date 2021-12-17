CUDA Stream Compaction and Radix Sort
========================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Kaan Erdogmus
  * [LinkedIn](https://linkedin.com/in/kaanberk), [personal website](https://kaan9.github.io)
* Tested on: Windows 10, i7-8850H @ 2.592GHz 24GB, Quadro P1000

## Project Overview
This project parallelized several sequential but associative array operations on the GPU. These operations, at first glance,
are highly sequential but can be made parallel by effectively breaking them up to smaller terms, calculating the results on the
subarrays and joining them by taking advantage of the associative nature of the operation.

The most fundamental of these operations is `scan`. `scan` is like the familiar `reduce` operation from functional languages but
also produces intermediary calculations that can be used for other algorithms. Such an operation implemented
in this project is `compact`, which removes elements from an array satisfying a specified predicate. The `scan`
operation can also be used to implement sorting, with `radix sort`, which is also implemented as a part of this project.

## Features Implemented
* Scan CPU implementation
* Compact CPU implementation
* Naive GPU Scan implementation `O(n log n)`
* Optimized Work-Efficient GPU Scan implementation `O(n)`
* GPU Stream Compaction
* Thrust's implementation of scan and compact (for benchmarking)
* Radix Sort
* UTF-8 decoding (work-in-progress)

## Scan


The implementation of scan is templated to allow for both int and byte input arrays (byte array used for UTF-8 decoding)

## Compact

## Radix


## UTF-8 encoding
UTF-8 encode can be performed using compact. Parallelize on the level of code-point (one kernel per code-point), with as little
divergence as possible. The kernel should read the code-point, determine how many bytes (1 to 4) are needed for its UTF-8
representation, always output 4 bytes, substituting 0xFFFF (an invalid sequence in unicode) for the extra bytes, then remove
these with compact to get a valid UTF-8 representation.

As an added bonus, the encoder can detect invalid characters and substitute the Unicode Replacement character � (U+FFFD)


## Test Output
(put output here for a large array size)

---------
Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

