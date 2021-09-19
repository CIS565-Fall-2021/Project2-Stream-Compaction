#include <cstdio>
#include <stdio.h>
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
            bool localStartCall = false;
            if (!timer().getCpuTimerStarted()) {
                timer().startCpuTimer();
                localStartCall = true;
            }
            
            odata[0] = 0; // identity for exclusive scan
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            
            if (localStartCall) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int numElts = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[numElts] = idata[i];
                    numElts++;
                }
            }
            timer().endCpuTimer();
            return numElts;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *boolArray = new int[n];
            int *scanResult = new int[n];
            int numElts = 0;

            for (int i = 0; i < n; i++) idata[i] != 0 ? boolArray[i] = 1 : boolArray[i] = 0;
            scan(n, scanResult, boolArray);
            for (int i = 0; i < n; i++) {
                if (boolArray[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                    numElts++;
                }
            }
            timer().endCpuTimer();
            return numElts;
        }
    }
}
