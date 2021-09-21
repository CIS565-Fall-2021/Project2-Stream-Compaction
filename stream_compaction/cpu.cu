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
            if (n == 0) return;
            timer().startCpuTimer();
            // exclusive scan 
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int k = 0;
            timer().startCpuTimer();
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[k++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return k;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *scanResults = new int[n];
            float time = 0;
            // timer starts after allocation
            timer().startCpuTimer();

            // mapping boolean function
            for (int i = 0; i < n; i++) {
                odata[i] = idata[i] != 0;
            }

            //scan
            scan(n, scanResults, odata);
            
            //compaction
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    k++;
                    odata[scanResults[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            printf("Work-Efficient scan: %f ms\n", time);
            return k;
        }
    }
}
