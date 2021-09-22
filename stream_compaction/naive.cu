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



    namespace Naive_Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Perform scan on arr
        // Use shared memory to reduce memory access latency
        // Notice that this can only process within ONE block, so n is at most as SAME as max number of threads in a block
        // 
        // Reference: GPU Gem Ch 39 Example 39.1
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // Bug in origin code
        __global__ void kern_prescan(int n, int *arr) {
            extern __shared__ int shared_buffer[];

            int index = threadIdx.x;
            int out_index = 0, in_index = 1;

            // Copy data to shared memory
            shared_buffer[index] = arr[index];

            __syncthreads();

            // Add pairs
            for (int offset = 1; offset < n; offset *= 2) {
                // Swap indices for two halves of array
                shared_buffer[in_index * n + index] = shared_buffer[out_index * n + index];
                out_index = 1 - out_index;
                in_index = 1 - out_index;

                if (index >= offset) {
                    shared_buffer[out_index * n + index] += shared_buffer[in_index * n + index - offset];
                }
                else {
                    shared_buffer[out_index * n + index] = shared_buffer[in_index * n + index];
                }

                // Synchronize all threads at each turn
                __syncthreads();
            }
            
            // Copy data back
            // Shift by 1 and set 0 for first element
            arr[index] = index > 0 ? shared_buffer[out_index * n + index - 1] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool timing_on) {
            // Create device array
            int *dev_array;
            cudaMalloc((void **)&dev_array, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");

            // Copy data to GPU
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed!");

            if (timing_on) {
                timer().startGpuTimer();
            }

            kern_prescan << <1, n, n * 2 * sizeof(int) >> > (n, dev_array);
            checkCUDAError("kern_prescan failed!");

            if (timing_on) {
                timer().endGpuTimer();
            }

            // Copy data back
            cudaMemcpy(odata, dev_array, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            // Cleanup
            cudaFree(dev_array);
            checkCUDAError("cudaFree failed!");
        }
    }
}
