#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        int* scan(int n, int *odata, int *idata);

        int compact(int n, int *odata, int *idata);
    }
}
