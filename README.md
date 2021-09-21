CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Matt Elser
  * [LinkedIn](https://www.linkedin.com/in/matt-elser-97b8151ba/), [twitter](twitter.com/__mattelser__ )
* Tested on: Tested on: Ubuntu 20.04, i3-10100F @ 3.6GHz 16GB, GeForce 1660 Super 6GB

### Main Features

### Time Comparison


### Known limitations
- [FIXED] The Naive implementation fails for array sizes greater than 2^25. 
  - Naive was calling an inefficient number of threads, leading to higher-than needed `threadIdx.x` 
    values. When multiplied to get the `index` this overflowed int and yielded a negative index.
    Logic around indices (reasonably) assumed positive values and therefore caused an out of bounds write.
- compact scan fails for array sizes greater than 2^27 due to running out of  CPU memory on the (16Gb) test machine.


Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

