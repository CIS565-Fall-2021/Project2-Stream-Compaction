#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        void optimizedScan(int n, int* odata, const int* idata);
        void optimizedScanRecursive(int n, int* dev_data);

        int compact(int n, int *odata, const int *idata);
        int optimizedCompact(int n, int* odata, const int* idata);

        __global__ void kernUpSweep(int n, int level, int* arr);
        __global__ void kernDownSweep(int n, int level, int* arr);
        __global__ void kernSetRootZero(int n, int* arr);
        __global__ void kernScanShared(int n, int logn, int* arr);
        __global__ void kernBlockIncrement(int n, int* data, int* increment);
    }
}
