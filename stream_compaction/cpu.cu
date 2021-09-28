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
            // TODO

            if (n <= 0) {
                return;
            }
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
            // TODO

            int writeIndex = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[writeIndex] = idata[i];
                    writeIndex++;

                }
            }
            timer().endCpuTimer();
            return writeIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* tempArr = new int[n];
            
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    tempArr[i] = 0;
                }
                else {
                    tempArr[i] = 1;
                }
            }

            int* scanned = new int[n];
            scanned[0] = 0;
            for (int i = 1; i < n; i++) {
                scanned[i] = scanned[i - 1] + tempArr[i - 1];
            }

            int result = 0;
            for (int i = 0; i < n; i++) {
                if (tempArr[i] == 1) {
                    odata[scanned[i]] = idata[i];
                    result++;
                }
            }
            delete[] tempArr;
            delete[] scanned;
            
            timer().endCpuTimer();
            return result;

            
        }
    }
}
