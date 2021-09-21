#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction
{
    namespace Naive
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernScanNaive(int n, int layer, int offset, int *odata, const int *idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            int tmp = idata[index];
            odata[index] = tmp + (index >= offset) * idata[index - offset];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            int power = ilog2ceil(n);
            int size = pow(2, power);
            int offset = size - n;
            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
            int *bufA;
            int *bufB;
            cudaMalloc((void **)&bufA, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc bufA failed!");
            cudaMalloc((void **)&bufB, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc bufB failed!");

            cudaMemset(bufA, 0, size * sizeof(int));
            cudaMemset(bufB, 0, size * sizeof(int));
            cudaMemcpy(bufA + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int *tmp;
            timer().startGpuTimer();
            // TODO
            for (int layer = 0; layer < power; layer++)
            {
                // invoke kernel
                int offset = pow(2, layer);
                kernScanNaive<<<fullBlocksPerGrid, blockSize>>>(size, layer, offset, bufB, bufA);
                cudaDeviceSynchronize();
                // swap bufA and bufB
                tmp = bufA;
                bufA = bufB;
                bufB = tmp;
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            cudaMemcpy(odata + 1, bufA + offset, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            cudaFree(bufA);
            checkCUDAErrorWithLine("cudaFree bufA failed!");
            cudaFree(bufB);
            checkCUDAErrorWithLine("cudaFree bufB failed!");
        }
    }
}
