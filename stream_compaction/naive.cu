#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

// Block size used for CUDA kernel launch
#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Add each value at (index-2^(d-1)) to the value at (index)
        __global__ void kern_add_pairs(int n, int d, const int* idata, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || index < (1 << (d - 1))) {
                return;
            }

            odata[index] = idata[index] + idata[index - (1 << (d - 1))];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Create device array and a buffer
            int *dev_array;
            int *dev_array_buf;
            cudaMalloc((void **)&dev_array, n * sizeof(int));
            cudaMalloc((void **)&dev_array_buf, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");

            // Copy data to GPU
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_array_buf, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed!");

            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();
            
            // Add for log(n) times
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kern_add_pairs << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, dev_array, dev_array_buf);
                checkCUDAError("kern_add_pairs failed!");

                // Ping-pong the buffers
                cudaMemcpy(dev_array, dev_array_buf, sizeof(int) * n, cudaMemcpyDeviceToDevice);
                checkCUDAError("ping-pong failed!");
            }

            // Set identity
            odata[0] = 0;

            timer().endGpuTimer();

            // Copy data back
            // Shift inclusive scan to exclusive scan           
            cudaMemcpy(odata + 1, dev_array, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            // Cleanup
            cudaFree(dev_array);
            cudaFree(dev_array_buf);
            checkCUDAError("cudaFree failed!");
        }
    }
}
