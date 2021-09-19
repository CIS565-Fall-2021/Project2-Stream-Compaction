#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

// Block size used for CUDA kernel launch
#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Add each value at (index+2^(d+1)-1) to the value at (index+2^d-1) in place
        __global__ void kern_reduction(int n, int d, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            // Only for multiple of 2^(d+1)
            if ((index & ((1 << (d + 1)) - 1)) == 0) {
                idata[index + (1 << (d + 1)) - 1] += idata[index + (1 << d) - 1];
            }
        }

        // Up-Sweep phase of efficient scan
        void up_sweep(int n, int* idata) {
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Reduction for log(n) times
            for (int d = 0; d < ilog2ceil(n); d++) {
                kern_reduction << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, idata);
                checkCUDAError("kern_reduction failed!");
            }
        }

        // Add each value at (index+2^(d+1)-1) to the value at (index+2^d-1) in place
        __global__ void kern_child_swap_add(int n, int d, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            // Only for multiple of 2^(d+1)
            if ((index & ((1 << (d + 1)) - 1)) == 0) {
                int temp = idata[index + (1 << d) - 1];
                idata[index + (1 << d) - 1] = idata[index + (1 << (d + 1)) - 1];
                idata[index + (1 << (d + 1)) - 1] += temp;
            }
        }

        // Set last element to zero
        __global__ void kern_clear_root(int n, int *idata) {
            idata[n - 1] = 0;
        }

        // Down-Sweep phase of efficient scan
        void down_sweep(int n, int* idata) {
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Set root to zero
            kern_clear_root << <1, 1 >> > (n, idata);
            checkCUDAError("kern_clear_root failed!");

            // log(n) passes
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kern_child_swap_add << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, idata);
                checkCUDAError("kern_child_swap_add failed!");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool timing_on) {
            // Create device array
            // Rounded to next power of two
            int round_n = 1 << ilog2ceil(n);
            int *dev_array;
            cudaMalloc((void**)&dev_array, round_n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");

            // Copy data to GPU
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed!");

            if (timing_on) {
                timer().startGpuTimer();
            }

            up_sweep(round_n, dev_array);

            down_sweep(round_n, dev_array);

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

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Used for computing the number of elements remaining after compaction
            int *last_elements = new int[2];

            // Create device array
            int *dev_array;
            int *dev_bool_buffer;
            int *dev_scan_buffer;
            int *dev_res;
            cudaMalloc((void **)&dev_array, n * sizeof(int));
            cudaMalloc((void **)&dev_bool_buffer, n * sizeof(int));
            cudaMalloc((void **)&dev_scan_buffer, n * sizeof(int));
            cudaMalloc((void **)&dev_res, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");

            // Copy data to GPU
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed!");

            timer().startGpuTimer();

            // Set 1 for non-zero elements
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_bool_buffer, dev_array);
            checkCUDAError("kernMapToBoolean failed!");

            // Scan
            scan(n, dev_scan_buffer, dev_bool_buffer, false);

            // Scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_res, dev_array, dev_bool_buffer, dev_scan_buffer);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            // Fetch last element of bool array and scan array respectively
            cudaMemcpy(last_elements, dev_bool_buffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(last_elements + 1, dev_scan_buffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            // Compute the number of elements remaining after compaction
            int num_element = last_elements[0] + last_elements[1];
            free(last_elements);

            // Copy data back
            cudaMemcpy(odata, dev_res, sizeof(int) * num_element, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            // Cleanup
            cudaFree(dev_array);
            cudaFree(dev_bool_buffer);
            cudaFree(dev_scan_buffer);
            cudaFree(dev_res);
            checkCUDAError("cudaFree failed!");

            return num_element;
        }
    }



    namespace Efficient_Upgraded {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Add each value at (index+2^(d+1)-1) to the value at (index+2^d-1) in place
        __global__ void kern_reduction(int n, int d, int *idata) {
            unsigned long int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // Index hack
            // Make use of all threads
            index *= (1 << (d + 1));

            if (index >= n) {
                return;
            }

            // 'index' is now multiple of 2^(d+1)
            idata[index + (1 << (d + 1)) - 1] += idata[index + (1 << d) - 1];
        }

        // Up-Sweep phase of efficient scan
        void up_sweep(int n, int *idata) {
            // Number of active elements in array
            int act_n = n;

            // Reduction for log(n) times
            for (int d = 0; d < ilog2ceil(n); d++) {
                // Halve the number of blocks launched in each turn
                act_n /= 2;
                dim3 fullBlocksPerGrid((act_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kern_reduction << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, idata);
                checkCUDAError("kern_reduction failed!");
            }
        }

        // Add each value at (index+2^(d+1)-1) to the value at (index+2^d-1) in place
        __global__ void kern_child_swap_add(int n, int d, int *idata) {
            unsigned long int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // Index hack
            // Make use of all threads
            index *= (1 << (d + 1));

            if (index >= n) {
                return;
            }

            // 'index' is now multiple of 2^(d+1)
            int temp = idata[index + (1 << d) - 1];
            idata[index + (1 << d) - 1] = idata[index + (1 << (d + 1)) - 1];
            idata[index + (1 << (d + 1)) - 1] += temp;
        }

        // Set last element to zero
        __global__ void kern_clear_root(int n, int *idata) {
            idata[n - 1] = 0;
        }

        // Down-Sweep phase of efficient scan
        void down_sweep(int n, int *idata) {
            // Set root to zero
            kern_clear_root << <1, 1 >> > (n, idata);
            checkCUDAError("kern_clear_root failed!");

            // Number of active elements in array
            int act_n = n / (1 << (ilog2ceil(n) + 1)) < 1 ? 1 : n / (1 << (ilog2ceil(n) + 1));

            // log(n) passes
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                // Double the number of blocks launched in each turn
                act_n *= 2;
                dim3 fullBlocksPerGrid((act_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kern_child_swap_add << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, idata);
                checkCUDAError("kern_child_swap_add failed!");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool timing_on) {
            // Create device array
            // Rounded to next power of two
            int round_n = 1 << ilog2ceil(n);
            int *dev_array;
            cudaMalloc((void **)&dev_array, round_n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");

            // Copy data to GPU
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed!");

            if (timing_on) {
                timer().startGpuTimer();
            }

            up_sweep(round_n, dev_array);

            down_sweep(round_n, dev_array);

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

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Used for computing the number of elements remaining after compaction
            int *last_elements = new int[2];

            // Create device array
            int *dev_array;
            int *dev_bool_buffer;
            int *dev_scan_buffer;
            int *dev_res;
            cudaMalloc((void **)&dev_array, n * sizeof(int));
            cudaMalloc((void **)&dev_bool_buffer, n * sizeof(int));
            cudaMalloc((void **)&dev_scan_buffer, n * sizeof(int));
            cudaMalloc((void **)&dev_res, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");

            // Copy data to GPU
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed!");

            timer().startGpuTimer();

            // Set 1 for non-zero elements
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_bool_buffer, dev_array);
            checkCUDAError("kernMapToBoolean failed!");

            // Scan
            scan(n, dev_scan_buffer, dev_bool_buffer, false);

            // Scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_res, dev_array, dev_bool_buffer, dev_scan_buffer);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            // Fetch last element of bool array and scan array respectively
            cudaMemcpy(last_elements, dev_bool_buffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(last_elements + 1, dev_scan_buffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            // Compute the number of elements remaining after compaction
            int num_element = last_elements[0] + last_elements[1];
            free(last_elements);

            // Copy data back
            cudaMemcpy(odata, dev_res, sizeof(int) * num_element, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            // Cleanup
            cudaFree(dev_array);
            cudaFree(dev_bool_buffer);
            cudaFree(dev_scan_buffer);
            cudaFree(dev_res);
            checkCUDAError("cudaFree failed!");

            return num_element;
        }
    }
}
