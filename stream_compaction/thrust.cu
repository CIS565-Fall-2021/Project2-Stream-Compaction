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
            // Create thrust vector and cast to GPU
            thrust::host_vector<int> thrust_idata(idata, idata + n);
            thrust::device_vector<int> dev_thrust_idata = thrust_idata;
            thrust::device_vector<int> dev_thrust_odata(n);
            
            timer().startGpuTimer();

            thrust::exclusive_scan(dev_thrust_idata.begin(), dev_thrust_idata.end(), dev_thrust_odata.begin());

            timer().endGpuTimer();

            int *dev_odata = thrust::raw_pointer_cast(dev_thrust_odata.data());
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");
        }
    }
}
