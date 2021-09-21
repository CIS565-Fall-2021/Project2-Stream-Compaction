/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <vector>
#include "testing_helpers.hpp"

const int SIZE = 1 << 16; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /*
    //For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); 
    */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    // --- repeated timing --- 
    int NUM_TIMINGS = 100;
    std::vector<float> data;
    float stdDev;
    float mean;

    printf("  Data gathered from %i runs with array size %i (2^%i)\n", NUM_TIMINGS, SIZE, ilog2(SIZE));
    printf("--------------------------------------------------------------\n\n");
    printf("------------------------------| mean (ms) |--| stdDev (ms) |--\n");
    printf("------ Scan ------\n");

    // CPU
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
        StreamCompaction::CPU::scan(SIZE, c, a);
        data.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("CPU Scan                \t%f\t%f\n", mean, stdDev);
    data.clear();

    // Naive
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
        StreamCompaction::Naive::scan(SIZE, c, a);
        data.push_back(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("Naive GPU Scan          \t%f\t%f\n", mean, stdDev);
    data.clear();

    // work efficient
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
        StreamCompaction::Efficient::scan(SIZE, c, a);
        data.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("Work Efficient GPU Scan \t%f\t%f\n", mean, stdDev);
    data.clear();
	
    // work efficient
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
		StreamCompaction::Thrust::scan(SIZE, c, a);
        data.push_back(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("Thrust Library Scan      \t%f\t%f\n", mean, stdDev);
    data.clear();

    printf("----- Compact -----\n");

    // CPU
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
        StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
        data.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("CPU compact without Scan \t%f\t%f\n", mean, stdDev);
    data.clear();
    
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
        StreamCompaction::CPU::compactWithScan(SIZE, b, a);
        data.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("CPU compact with Scan    \t%f\t%f\n", mean, stdDev);
    data.clear();
    
    // work efficient
    for (int i = 0; i < NUM_TIMINGS; i++) {
        zeroArray(SIZE, c);
        StreamCompaction::Efficient::compact(SIZE, c, a);
        data.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    }
    tabulate(&data, &mean, &stdDev);
    printf("Work Efficient GPU compact\t%f\t%f\n", mean, stdDev);
    data.clear();
	
    std::cout << std::endl;

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
