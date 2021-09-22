#include <cstdio>
#include "cpu.h"
#include "testing_helpers.hpp"
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
            /*
             * out[0] = in[0]; // assuming n > 0
               for (int k = 1; k < n; ++k)
               out[k] = out[k – 1] + in[k];
             */
            if(n < 1) return;

            // inclusive
            odata[0] = idata[0];
            for(int i=1; i<n; ++i){
              odata[i] = odata[i-1] + idata[i];
            }

            // make exclusive
//						for(int i=n-1; i>0; --i){
//							odata[i] = odata[i-1];
//						}
//						odata[0] = 0;

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int numNonZero = 0;
            for(int i=0; i<n; ++i){
              if(idata[i] != 0){
                odata[numNonZero] = idata[i];
                numNonZero++;
              }
            }

            timer().endCpuTimer();
            return numNonZero;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            if(n < 1) return 0;

            int numNonZero = 0;
            int tmp[n];
            for(int i=0; i<n; ++i){
              if(idata[i] == 0){
                tmp[i] = 0;
              }else{
                tmp[i] = 1;
                numNonZero++;
              }
            }


            int scan[n];
            scan[0] = 0;
            for(int i=1; i<n; ++i){
              scan[i] = scan[i-1] + tmp[i-1];
            }


            for(int i=0; i<n; ++i) {
              if (tmp[i] == 1) {
                odata[scan[i]] = idata[i];
              }
            }

            timer().endCpuTimer();
            return numNonZero;
        }
    }
}
