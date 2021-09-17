#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();




            // TODO
//
//#pragma region NonPower2
//
//            int arr_z[400];
//            int arr_b[400];
//            int arr_c[400]; // exclusive array
//
//            int power2 = 1;
//            int nearesttwo = 2;
//
//            for (int i = 0; i < 5; i++)
//            {
//                nearesttwo = nearesttwo << 1;
//                if (nearesttwo >= n)
//                {
//                    break;
//                }
//            }
//
//
//            int difference = nearesttwo - n;
//
//            for (int i = 0; i < difference; i++)
//            {
//                arr_z[difference] = 0;
//            }
//
//
//            for (int i = 0; i < n; i++)
//            {
//                arr_z[i + difference] = idata[i];
//            }
//
//            n = n + difference;
//
//            for (int i = 0; i < n; i++)
//            {
//                if (arr_z[i] == 0)
//                {
//                    arr_b[i] = 0;
//                    continue;
//                }
//                arr_b[i] = 1;
//            }
//
//            arr_c[0] = 0;
//            for (int i = 1; i < n; i++)
//            {
//                arr_c[i] = arr_b[i - 1] + arr_c[i - 1];
//            }
//
//            for (int i = 0; i < n; i++)
//            {
//                if (arr_b[i] == 0)
//                {
//                    continue;
//                }
//                int index = arr_c[i];
//                odata[index] = idata[i];
//            }
//
//
//
//#pragma endregion

            timer().endGpuTimer();
        }
    }
}
