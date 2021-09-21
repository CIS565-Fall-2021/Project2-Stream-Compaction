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
            // TODO (DONE)
            odata[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                odata[i + 1] = idata[i] + odata[i];
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
            // TODO (DONE)
            int num = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[num] = idata[i];
                    num++;
                }
            }
            timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int *arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = idata[i] ? 1 : 0;
            }

            int* scanArr = new int[n];
            //scan(n, scanArr, arr);
            scanArr[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                scanArr[i + 1] = arr[i] + scanArr[i];
            }

            int num = 0;
            for (int i = 0; i < n; i++) {
                if (arr[i]) {
                    odata[scanArr[i]] = idata[i];
                    num++;
                }
            }
            timer().endCpuTimer();
            return num;
        }
    }
}
