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
            // tally a running sum of input data 
            int sum = 0;
            for (int i = 0; i < n; i++) {
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
            // TODO
            // if condition is met, scan and scatter at the same time (sort of) 
            int index = 0; 
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[index++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            if (n < 1) { return 0;  }

            // boolean buffer
            int* ibool = (int*) malloc(n * sizeof(int));

            // map input array to boolean
            for (int i = 0; i < n; i++) {
                ibool[i] = idata[i] != 0; 
            }

            // scan boolean buffer
            // memory error is thrown when calling StreamCompaction::CPU::scan()
            int iboolScan = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = iboolScan;
                iboolScan += ibool[i]; 
            }

            int numElements = odata[n - 1] + ibool[n - 1]; 

            // scatter 
            for (int i = 0; i < n; i++) {
                if (ibool[i]) {
                    odata[odata[i]] = idata[i]; 
                }
            }

            free(ibool); 

            timer().endCpuTimer();
            return numElements;
        }
    }
}
