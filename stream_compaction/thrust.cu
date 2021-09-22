#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

#define checkCUDAErrorWithLine(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)
#define blockSize 128

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
        void scan(int N, int *odata, const int *idata) {

        	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

						// copy data to gpu buffer
						int* dev_odata;
						int* dev_idata;
						cudaMalloc((void**)&dev_odata, N * sizeof(int));
						checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
						cudaMalloc((void**)&dev_idata, N * sizeof(int));
						checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

						cudaMemcpy(dev_odata, odata, sizeof(int) * N, cudaMemcpyHostToDevice);
						cudaMemcpy(dev_idata, idata, sizeof(int) * N, cudaMemcpyHostToDevice);


						timer().startGpuTimer();

						thrust::device_ptr<int> dev_thrust_odata(dev_odata);
						thrust::device_ptr<int> dev_thrust_idata(dev_idata);

						 // TODO use `thrust::exclusive_scan`
						            // example: for device_vectors dv_in and dv_out:
						            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
						thrust::inclusive_scan(dev_thrust_idata, dev_thrust_idata+N, dev_thrust_odata);

						//cudaDeviceSynchronize();
						timer().endGpuTimer();

						cudaMemcpy(odata, dev_odata, sizeof(int) * N, cudaMemcpyDeviceToHost);

						cudaDeviceSynchronize();
						cudaFree(dev_odata);
						cudaFree(dev_idata);

        }
    }
}
