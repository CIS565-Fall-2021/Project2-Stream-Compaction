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




#### Decoder
For an input size of `n`, the decoder performs `O(n)` total operations.


The current design could be extended to error handling. Specifically, we could have a kernel spawned per byte of input,
which analyze the byte to check if it is at the start of a code-point and if so, check if the following bytes complete
the code-point in a valid way and if that's the case, write a 1 to a corresponding array of bools for each of these
bytes. Then, we could invert the array of bools and using it as a key, run compact on the input array to remove the
bytes that are not part of valid code-points, producing a valid UTF-8 array. The difference in length between the
original and compacted array gives the number of invalid bytes removed.

Alternatively, it might be more desirable to explicitly show the incorrect bytes in the output, for example by
substituting the Unicode Replacement Character � (U+FFFD). This can be accomplished by first determining all invalid
bytes in the input as described above, then selecting the expansion offset as +4 similar to the original algorithm,
adn then modifying the subsequent kernel call that compacts the bits so that it checks for invalid input and
substitutes the U+FFFD character.

## Test Output
N = 2^24
```
****************
** SCAN TESTS **                                                                                                        ****************
[  40  24   0  29  16  10   6   0  10  10   7  10  15 ...  38   0 ]
==== cpu scan, power-of-two ====
elapsed time: 54.4685ms    (std::chrono Measured)
[   0  40  64  64  93 109 119 125 125 135 145 152 162 ... 386716816 386716854 ]
==== cpu scan, non-power-of-two ====                                                                                       
elapsed time: 67.3799ms    (std::chrono Measured)                                                                       
[   0  40  64  64  93 109 119 125 125 135 145 152 162 ... 386716768 386716800 ]                                         
passed                                                                                                              
==== naive scan, power-of-two ====                                                                                         
elapsed time: 197.895ms    (CUDA Measured)                                                                               
passed                                                                                                              
==== naive scan, non-power-of-two ====                                                                                     
elapsed time: 197.504ms    (CUDA Measured)                                                                               
passed                                                                                                              
==== work-efficient scan, power-of-two ====                                                                                
elapsed time: 74.3567ms    (CUDA Measured)                                                                               
passed                                                                                                              
==== work-efficient scan, non-power-of-two ====                                                                            
elapsed time: 74.111ms    (CUDA Measured)                                                                                
passed                                                                                                              
==== thrust scan, power-of-two ====                                                                                        
elapsed time: 2229.07ms    (CUDA Measured)                                                                               
passed                                                                                                              
==== thrust scan, non-power-of-two ====                                                                                    
elapsed time: 2224.75ms    (CUDA Measured)                                                                               
passed                                                                                                                              
*****************************                                                                                           
** STREAM COMPACTION TESTS **   
*****************************                                                                                               
[   0   1   2   0   0   2   2   3   0   0   0   0   3 ...   0   0 ]                                                 
==== cpu compact without scan, power-of-two ====                                                                           
elapsed time: 167.942ms    (std::chrono Measured)                                                                        
[   1   2   2   2   3   3   2   1   3   2   2   1   2 ...   3   2 ]                                                     
passed                                                                                                              
==== cpu compact without scan, non-power-of-two ====                                                                       
elapsed time: 267.453ms    (std::chrono Measured)                                                                        
[   1   2   2   2   3   3   2   1   3   2   2   1   2 ...   1   3 ]                                                     
passed                                                                                                              
==== cpu compact with scan ====                                                                                            
elapsed time: 364.678ms    (std::chrono Measured)                                                                        
[   1   2   2   2   3   3   2   1   3   2   2   1   2 ...   3   2 ]                                                     
passed                                                                                                              
==== work-efficient compact, power-of-two ====                                                                             
elapsed time: 92.2182ms    (CUDA Measured)                                                                               
passed                                                                                                              
==== work-efficient compact, non-power-of-two ====                                                                         
elapsed time: 91.914ms    (CUDA Measured)                                                                                
passed                                                                                                              
==== thrust compact, power-of-two ====                                                                                     
elapsed time: 91.914ms    (CUDA Measured)                                                                                
passed                                                                                                              
==== work-efficient compact, non-power-of-two ====                                                                         
elapsed time: 91.914ms    (CUDA Measured)                                                                                
passed                   

*****************************                                                                                           
** RADIX SORT TEST **                                                                                                 
*****************************    
[ 2176510 238971848 427268424 345874100 22071882 53876664 93690960 91803616 16555815 541773164 18448217 67772704 508518232 ... 32860010 272354542 ]
==== cpu radix using std::sort, power of two ====                                                                          
elapsed time: 13524.7ms    (std::chrono Measured)                                                                        
[   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 1072890000 1072988290 ]                                   
==== cpu radix using std::sort, non-power of two ====                                                                      
elapsed time: 12362.3ms    (std::chrono Measured)                                                                        
[   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 1072890000 1072988290 ]                                   
==== gpu radix, power of two ====                                                                                         
elapsed time: 3073.04ms    (CUDA Measured)                                                                               
[   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 1072890000 1072988290 ]                                
passed                                                                                                        
==== gpu radix, non-power of two ====                                                                        
elapsed time: 3074.25ms    (CUDA Measured)                                                                  
[   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 1072890000 1072988290 ]                        
passed

```
Notes
---------
CMakeLists.txt has been edited to add the new dependencies `radix_sort.h`, `radix_sort.cu`, `utf-8.h`, and `utf-8.cu`.

---------
Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

