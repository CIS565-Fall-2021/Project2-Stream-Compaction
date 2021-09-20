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

        __global__ void kernScan(int n, int d, const int* idata, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int pow2d = 1 << (d - 1);
            if (index >= pow2d) {
                odata[index] = idata[index] + idata[index - pow2d];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* deviceIn;
            int* deviceOut;
            cudaMalloc((void**)&deviceIn, n * sizeof(int));
            cudaMalloc((void**)&deviceOut, n * sizeof(int));
            cudaMemcpy(deviceIn, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(deviceOut, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            int blockSize = 128;
           
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan << < fullBlocksPerGrid, blockSize >> > (n, d, deviceIn, deviceOut);
                checkCUDAError("kernScan failed");
                cudaMemcpy(deviceIn, deviceOut, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }
            timer().endGpuTimer();

            odata[0] = 0;
         
            //shift
            cudaMemcpy(odata + 1, deviceIn, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);

            checkCUDAError("cudaMemcpy back failed!");
            cudaFree(deviceIn);
            cudaFree(deviceOut);
        }
    }
}
