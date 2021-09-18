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
            // DONE
            // Exclusive scan includes 0 at first.
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
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
            // DONE
            int size = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[size++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return size;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // DONE

            // Bit
            for (int i = 0; i < n; ++i) {
                odata[i] = idata[i] != 0 ? 1 : 0;
            }

            // Scan
            int size = 0;
            for (int i = 0; i < n; ++i) {
                int temp = odata[i];
                odata[i] = size;
                size += temp;
            }

            // Scatter.
            for (int i = 0; i < n; ++i) {
                odata[odata[i]] = idata[i];
            }

            timer().endCpuTimer();
            return size;
        }
    }
}
