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

            odata[0] = idata[0];
            for (int i = 1; i < n; i++) {
              odata[i] = odata[i - 1] + idata[i];
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
            for (int i = 0; i < n; i++) {
              if (idata[i] > 0) {
                odata[j++] = idata[i];
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
          
          // criteria array
          int* tmp = new int[n * sizeof(int)];
          int* scan_out = new int[n * sizeof(int)];
          
          timer().startCpuTimer();

          for (int i = 0; i < n; i++) {
            tmp[i] = (idata[i] > 0) ? 1 : 0;
          }

          // inclusive scan
          scan_out[0] = tmp[0];
          for (int i = 1; i < n; i++) {
            scan_out[i] = scan_out[i-1] + tmp[i];
          }
          int N = scan_out[n - 1];  // total number

          // make exclusive
          // shift array right
          for (int i = n; i > 0; i--) {
            scan_out[i] = scan_out[i-1];
          }
          scan_out[0] = 0;  // insert identity
          
          // scatter

          for (int i = 0; i < n; i++) {
            if (tmp[i] > 0) {
              odata[scan_out[i]] = idata[i];
            }
          }

          timer().endCpuTimer();

          free(tmp);
          free(scan_out);

          return N;
        }
    }
}
