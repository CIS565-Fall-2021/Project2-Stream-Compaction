#include <cstdio>
#include "cpu.h"

#include "common.h"

#include <iostream>
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
            timer().startCpuTimer();
            int o = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[o] = idata[i];
                    o++;
                }
            }
            timer().endCpuTimer();
            return o;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* mask = new int[n];
            for (int i = 0; i < n; i++) {
                mask[i] = idata[i] == 0 ? 0 : 1;
            }

            int* scatter = new int[n];

            scatter[0] = 0;
            for (int i = 1; i < n; i++) {
              scatter[i] = scatter[i - 1] + mask[i - 1];
            }

            int o = 0;
            for (int i = 0; i < n; i++) {
              if (mask[i] != 0) {
                odata[scatter[i]] = idata[i];
                o++;
              }
            }
            timer().endCpuTimer();
            return o;
        }
    }
}
