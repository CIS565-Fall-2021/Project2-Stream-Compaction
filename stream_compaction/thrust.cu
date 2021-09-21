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
            int* dev_odata;
            int* dev_idata;
            
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_odata!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_idata!");

            thrust::host_vector<int> host_thrust_odata(odata, odata + n);
            thrust::device_vector<int> dev_thrust_odata = (thrust::device_vector<int>) host_thrust_odata;
            thrust::host_vector<int> host_thrust_idata(idata, idata + n);
            thrust::device_vector<int> dev_thrust_idata = (thrust::device_vector<int>) host_thrust_idata;

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_idata.begin(), dev_thrust_idata.end(), dev_thrust_odata.begin());
            timer().endGpuTimer();

            host_thrust_odata = (thrust::host_vector<int>) dev_thrust_odata;
            thrust::copy(host_thrust_odata.begin(), host_thrust_odata.end(), odata);

            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree failed on dev_odata!");
            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree failed on dev_idata!");
        }
    }
}
