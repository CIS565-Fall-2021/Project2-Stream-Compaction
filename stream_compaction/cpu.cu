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
            odata[0] = 0; // explicit starts with 0
            for (int i = 0; i < n - 1; i++){
                odata[i+1] = odata[i] + idata[i];   
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
            
            int compactSize = 0;
            for (int i = 0; i < n; i++){
                bool elem = idata[i];
                if (elem){
                    odata[compactSize] = elem; 
                    compactSize++;
                }
            }
            
            timer().endCpuTimer();
            return compactSize;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // map bools to 0s and 1s
            int intData[n];
            for (int i = 0; i < n; i++){
                intData[i] = idata[i] ? 1 : 0;   
            }
            
            // scan resulting array
            int scannedData[n];
            scan(n, &scannedData[0], &intData[0]);
            
            // use scatter to produce output
            int size = scannedData[n-1] - 1;
            int scatter_i = 0;
            for (int i = 0; i < n; i++){
                if (idata[i]){ 
                    odata[scatter_i] = idata[i]; 
                    scatter_i++;
                }    
            }
            
            timer().endCpuTimer();
            return size;
        }
    }
}
