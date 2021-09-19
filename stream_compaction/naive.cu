#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernIterateScan(int n, int d, int* srcBuffer, int* desBuffer) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;

          if (n <= index) {
            return;
          }

          int p = 1 << d - 1;
          if (index < n) {
            desBuffer[index] = srcBuffer[index] + (p <= index ? srcBuffer[index - p] : 0);
          }
        }

        // This is totally amateurish, but hey, this *is* supposed to be the "naive" scan!
        __global__ void kernConvertInclusiveToExclusive(int n, int* srcBuffer, int* desBuffer) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;

          if (index && index < n) {
            desBuffer[index] = srcBuffer[index-1];
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          int paddedN = int(pow(2, ilog2ceil(n)));

          int blockSize = 128;
          int gridSize = ceil(paddedN * 1.0 / blockSize);
          int* device_flipflopA;
          int* device_flipflopB;

          int* device_srcBuffer;
          int* device_desBuffer;

          cudaMalloc((void**)&device_flipflopA, paddedN * sizeof(int));
          checkCUDAError("cudaMalloc dev_A failed!");

          cudaMalloc((void**)&device_flipflopB, paddedN * sizeof(int));
          checkCUDAError("cudaMalloc dev_B failed!");

          cudaMemcpy(device_flipflopA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          cudaDeviceSynchronize();

          timer().startGpuTimer();
          for (int d = 1; d <= ilog2ceil(n); d++) {
            device_srcBuffer = d % 2 == 0 ? device_flipflopB : device_flipflopA;
            device_desBuffer = d % 2 == 0 ? device_flipflopA : device_flipflopB;
            kernIterateScan << <gridSize, blockSize >> > (n, d, device_srcBuffer, device_desBuffer);
            cudaDeviceSynchronize();
            checkCUDAError("Iteration error");
          }

          kernConvertInclusiveToExclusive << <gridSize, blockSize >> > (n, device_desBuffer, device_srcBuffer);
          cudaDeviceSynchronize();
          timer().endGpuTimer();

          cudaMemcpy(odata, device_srcBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();

          cudaFree(device_flipflopA);
          cudaFree(device_flipflopB);

          odata[0] = 0;
        }
    }
}
