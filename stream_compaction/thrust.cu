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
            thrust::host_vector<int> host(n);
            for (int i = 0; i < n; i++) {
                host[i] = idata[i];
            }
            thrust::device_vector<int> device = host;
            timer().startGpuTimer();
            thrust::exclusive_scan(device.begin(), device.end(), device.begin());
            timer().endGpuTimer();
            for (int i = 0; i < device.size(); i++) {
                odata[i] = device[i];
            }
        }
    }
}
