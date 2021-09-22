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
            int* dev_in;
            cudaMalloc((void**)&dev_in, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_in failed!");

            int* dev_out;
            cudaMalloc((void**)&dev_out, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_out failed!");

            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            thrust::device_ptr<int> dev_thrust_in = thrust::device_pointer_cast(dev_in);
            thrust::device_ptr<int> dev_thrust_out = thrust::device_pointer_cast(dev_out);

            timer().startGpuTimer();

            thrust::exclusive_scan(dev_thrust_in, dev_thrust_in + n, dev_thrust_out);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from device failed!");

            cudaFree(dev_in);
            checkCUDAError("cudaFree dev_in failed!");

            cudaFree(dev_out);
            checkCUDAError("cudaFree dev_out failed!");
        }
    }
}
