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
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = 0;

            for (int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int oi = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[oi] = idata[i];
                    oi++;
                }
            }
            timer().endCpuTimer();
            return oi;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int* shouldInclude = new int[n];
            int* scan = new int[n];

            for (int i = 0; i < n; i++) {
                shouldInclude[i] = (idata[i] != 0) ? 1 : 0;
            }

            scan[0] = 0;
            for (int i = 1; i < n ; i++) {
                scan[i] = shouldInclude[i-1] + scan[i - 1];
            }
            
            int lastIndex = 0;
            for (int i = 0; i < n; i++) {
                if (shouldInclude[i] != 0) {
                    lastIndex = scan[i];
                    odata[lastIndex] = idata[i];
                }
            }
            delete[] shouldInclude;
            delete[] scan;
            timer().endCpuTimer();
            return lastIndex+1;
        }
    }
}
