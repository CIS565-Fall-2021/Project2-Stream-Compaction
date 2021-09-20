#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool timing_on);

        int compact(int n, int *odata, const int *idata);
    }



    namespace Efficient_Upgraded {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, bool timing_on);
    }



    namespace Efficient_Shared {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, bool timing_on);
    }
}
