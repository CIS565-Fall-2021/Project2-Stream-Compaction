#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 32

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernScan(int offset, int n, int *dev_odata, int *dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            if (index >= offset) {
                dev_idata[index] = dev_odata[index - offset] + dev_odata[index];
            }
            else {
                dev_idata[index] = dev_odata[index];
            }
        }       

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int offset, *dev_odata, *dev_idata;
            // malloc memory before timing
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
	        cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int D = ilog2ceil(n);
            timer().startGpuTimer();
            // calling kernel function in for loop, will be executed in parallel
            for (int d=1; d<=D;d++){
                offset = 1 << (d - 1);
		        kernScan << <fullBlocksPerGrid, blockSize>> >(offset, n, dev_odata, dev_idata);
                // ping pong buffer
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            // printf("Naive scan: %f ms\n", timer().getGpuElapsedTimeForPreviousOperation());
            cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
	        odata[0] = 0;
            // free memory
            cudaFree(dev_odata);
	        cudaFree(dev_idata);
        }
    }
}
