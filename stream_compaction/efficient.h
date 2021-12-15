#pragma once

#include "common.h"
#include "cVec.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

	/* in-place scan over device array, doesn't start GPU Timer and assumes input is power of 2 */
	void scan_dev(int N, cu::cVec<int>* dev_data);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
