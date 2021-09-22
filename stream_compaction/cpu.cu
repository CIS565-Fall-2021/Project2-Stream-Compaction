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
            int curr;
            int sum = idata[0];
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                curr = idata[i];
                odata[i] = sum;
                sum += curr;
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
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* indicator = new int[n];
            timer().startCpuTimer();            
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    indicator[i] = 0;
                }
                else {
                    indicator[i] = 1;
                }
            }            
            odata[0] = 0;   // odata is currently the array storing the exclusive scan result 
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + indicator[i - 1];
            }
            int count = odata[n - 1] + indicator[n - 1];
            for (int i = 0; i < n; i++) {
                if (indicator[i] != 0) {
                    odata[odata[i]] = idata[i];     //odata is now the compacted array
                }
            }
            timer().endCpuTimer();
            return count;
        }
    }
}
