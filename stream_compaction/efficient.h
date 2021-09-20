#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        void scanBatch(int n, int* odata, const int* idata, bool splitOnce = true);

        int compactBatch(int n, int *odata, const int *idata, bool splitOnce = true);

        void sort(int n, int* odata, const int* idata);
    }
    namespace EfficientTest {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
    }
}
