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

            //for (int ind = 0; ind < n / 2; ind++) {
            //    printf("%d ", idata[ind]);
            //}
            //printf("\n");
            //printf("\n");
            //for (int ind = n / 2; ind < n; ind++) {
            //    printf("%d ", idata[ind]);
            //}
            //printf("\n");
            //printf("\n");
            

            timer().startCpuTimer();
            odata[0] = 0;
            for (int ind = 1; ind < n; ind++) {
                odata[ind] = idata[ind-1] + odata[ind - 1];
            }

            timer().endCpuTimer();

            //for (int ind = 0; ind < n / 2; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");
            //for (int ind = n / 2; ind < n; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int ind = 0; ind < n; ind++) {
                if (idata[ind] != 0) {
                    odata[count] = idata[ind];
                    count++;
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
            timer().startCpuTimer();
            
            int* bool_list = new int[n];
            int* scan_list = new int[n];
            for (int ind = 0; ind < n; ind++) {
                if (idata[ind] == 0) {
                    bool_list[ind] = 0;
                }
                else {
                    bool_list[ind] = 1;
                }
                //printf("%d ", bool_list[ind]);
            }
            //printf("\n");
            scan_list[0] = 0;
            //printf("%d ", scan_list[0]);
            for (int ind = 1; ind < n; ind++) {
                scan_list[ind] = bool_list[ind - 1] + scan_list[ind - 1];
                //printf("%d ", scan_list[ind]);
            }
            //printf("\n");
            int count = 0;
            for (int ind = 0; ind < n; ind++) {
                //printf("%d ", idata[ind]);
                if (bool_list[ind] == 1) {
                    odata[scan_list[ind]] = idata[ind];
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }
    }
}
