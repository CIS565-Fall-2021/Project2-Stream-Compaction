#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernel_naive_parallel_scan(const int* read, int* write, int d, int n)
        {
          int k = blockIdx.x * blockDim.x + threadIdx.x;

          if (k > n - 1)
            return;

          int step = (int)(powf(2, d - 1) + 0.5);
          if (k >= step)
            write[k] = read[k - step] + read[k];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // allocate memory for the read write buffers
            int* devRead, int* devWrite;
            cudaMalloc((void**)&devRead, n * sizeof(int));
            cudaMalloc((void**)&devWrite, n * sizeof(int));

            // Copy idata to read and write buffer
            cudaMemcpy(devRead, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(devWrite, devRead, n * sizeof(int), cudaMemcpyDeviceToDevice);
            
            // define kernel dimension
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // run naive scan
            for (int d = 1; d <= ilog2ceil(n); d++)
            {
              kernel_naive_parallel_scan<<<fullBlocksPerGrid, blockSize>>>(devRead, devWrite, d, n);

              // swap read and write
              cudaMemcpy(devRead, devWrite, n * sizeof(int), cudaMemcpyDeviceToDevice); // TODO: is there another way?
            }

            // Copy write buffer to odata
            odata[0] = 0;
            cudaMemcpy(odata + 1, devRead, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(devRead);
            cudaFree(devWrite);
            timer().endGpuTimer();
        }
    }
}
