#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        __global__ void kernParallelScan(int n, int level, int* src, int* dest);
        __global__ void kernInclusiveToExclusive(int n, int* src, int* dest);
    }
}
