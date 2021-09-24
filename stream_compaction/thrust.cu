#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          thrust::host_vector<int> thrustHostVector(idata, idata+n);
          thrust::device_vector<int> thrustDeviceVector = thrustHostVector;

          timer().startGpuTimer();
          thrust::exclusive_scan(thrustDeviceVector.begin(), thrustDeviceVector.end(), thrustDeviceVector.begin());
          timer().endGpuTimer();

          // This came from Stack Overflow.
          // There is undoubtedly a better way to do this.
          int i = 0;
          for (thrust::device_vector<int>::iterator iter = thrustDeviceVector.begin(); iter != thrustDeviceVector.end(); iter++) {
            odata[i++] = *iter;
          }
        }
    }
}
