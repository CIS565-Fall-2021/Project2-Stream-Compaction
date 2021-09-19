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
            odata[0] = 0; // identity
            for (int i = 1; i < n; i++)
              odata[i] = odata[i - 1] + idata[i - 1];
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int num = 0;
            for (int i = 0; i < n; i++)
            {
              if (idata[i] > 0)
                odata[num++] = idata[i];
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
            // Temporary array with 0 or 1 depending if entry is a nonzero value
            int* bitData = (int*) malloc(sizeof(int) * n);
            for (int i = 0; i < n; i++)
            {
              bitData[i] = (idata[i] > 0) ? 1 : 0;
            }
            
            // run exclusive scan on temporary array
            int* scannedBitData = (int*) malloc(sizeof(int) * n);
            scannedBitData[0] = 0; // identity
            for (int i = 1; i < n; i++)
              scannedBitData[i] = scannedBitData[i - 1] + bitData[i - 1];

            // scatter to compute the stream compaction
            for (int i = 0; i < n; i++)
            {
              if (bitData[i] == 1)
              {
                odata[scannedBitData[i]] = idata[i];
              }
            }

            // size of final array
            int num = ((bitData[n - 1] == 1) ? scannedBitData[n - 1] : scannedBitData[n - 2]) + 1;

            // free allocated memory
            free(bitData);
            free(scannedBitData);
            return num;
        }
    }
}
