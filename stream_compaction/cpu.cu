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

            int j = 0;
            for (int i = 0; i < n; ++i) {
              if (idata[i] != 0) {
                odata[j] = idata[i];
                ++j;
              }
            }

            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // map array to 0s and 1s
            int* bitmap = (int*)std::malloc(n * sizeof(int));
            for (int i = 0; i < n; ++i) {
              bitmap[i] = idata[i] != 0;
            }

            // scan implementation
            int* scannedBitmap = (int*)std::malloc(n * sizeof(int));
            int count = 0;
            for (int i = 0; i < n; ++i) {
              scannedBitmap[i] = count;
              count += bitmap[i];
            }

            for (int i = 0; i < n - 1; ++i) {
              if (scannedBitmap[i] != scannedBitmap[i + 1]) {
                odata[scannedBitmap[i]] = idata[i];
              }
            }

            std::free(bitmap);
            std::free(scannedBitmap);

            timer().endCpuTimer();

            return count;
        }
    }
}
