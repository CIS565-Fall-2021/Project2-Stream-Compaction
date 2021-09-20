#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (exclusive prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int j = 1; j < n; j++)
            {
                odata[j] = odata[j - 1] + idata[j - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         * This stream compaction method will remove 0s from an array of ints.
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // Given an array of elements, create a new array with all the 0s 
            // removed while preserving order
            int index = 0;
            for (int i = 0; i < n; i++)
            {
                int thisElement = idata[i];
                if (thisElement != 0)
                {
                    odata[index] = thisElement;
                    index++;
                }
            }
            timer().endCpuTimer();
            return n - index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         * This stream compaction method will remove 0s from an array of ints.
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            timer().endCpuTimer();
            return -1;
        }
    }
}
