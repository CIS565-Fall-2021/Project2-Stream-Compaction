#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs an in-place up sweep on the data.
         */
        __global__ void kernUpSweep(int N, int d, int *data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N)
            {
                return;
            }

            if (index % (1 << (d + 1)) == 0)
            {
                data[index + (1 << (d + 1)) - 1] += data[index + (1 << d) - 1];
            }
        }

        /**
         * Performs an in-place down sweep on the data.
         */
        __global__ void kernDownSweep(int N, int d, int *data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N)
            {
                return;
            }

            if (index % (1 << (d + 1)) == 0)
            {
                int t = data[index + (1 << d) - 1];
                data[index + (1 << d) - 1] = data[index + (1 << (d + 1)) - 1];
                data[index + (1 << (d + 1)) - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // Done - Part 3.1
            int tree_depth = ilog2ceil(n);
            int data_length = 1 << tree_depth;
            int data_bytes = data_length * sizeof(int);

            int* dev_data;
            cudaMalloc((void**)&dev_data, data_bytes);
            cudaMemset(dev_data, 0, data_bytes);
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((data_length + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Perform up sweep
            for (int d = 0; d <= tree_depth - 1; d++)
            {
                kernUpSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(data_length, d, dev_data);
            }

            // Zero out the root
            cudaMemset(dev_data + data_length - 1, 0, sizeof(int));

            // Perform down sweep
            for (int d = tree_depth - 1; d >= 0; d--)
            {
                kernDownSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(data_length, d, dev_data);
            }

            timer().endGpuTimer();

            // Copy scanned array to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
            // Done-Part 3.2
            int tree_depth = ilog2ceil(n);
            int data_length = 1 << tree_depth;
            int data_bytes = data_length * sizeof(int);
            int num_elts = 0;

            int *dev_orig_data;
            cudaMalloc((void**)&dev_orig_data, data_bytes);
            cudaMemset(dev_orig_data, 0, data_bytes);
            cudaMemcpy(dev_orig_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int *dev_output;
            cudaMalloc((void**)&dev_output, n * sizeof(int));

            int *dev_valid_indices;
            cudaMalloc((void**)&dev_valid_indices, data_bytes);

            int *dev_scanned_indices;
            cudaMalloc((void**)&dev_scanned_indices, data_bytes);

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((data_length + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Transform input into binary array
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>>(data_length, dev_valid_indices, dev_orig_data);
            cudaMemcpy(dev_scanned_indices, dev_valid_indices, data_bytes, cudaMemcpyDeviceToDevice);

            // Perform up sweep
            for (int d = 0; d <= tree_depth - 1; d++)
            {
                kernUpSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(data_length, d, dev_scanned_indices);
            }

            // Zero out the root
            cudaMemset(dev_scanned_indices + data_length - 1, 0, sizeof(int));

            // Perform down sweep
            for (int d = tree_depth - 1; d >= 0; d--)
            {
                kernDownSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(data_length, d, dev_scanned_indices);
            }

            // Write valid data to output based on indices computed in scan (scatter)
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, BLOCK_SIZE>>>(data_length, dev_output, dev_orig_data, dev_valid_indices, dev_scanned_indices);

            timer().endGpuTimer();

            // Copy filtered data to host
            cudaMemcpy(odata, dev_output, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Copy number of elements
            cudaMemcpy(&num_elts, dev_scanned_indices + data_length - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // Free all buffers
            cudaFree(dev_orig_data);
            cudaFree(dev_valid_indices);
            cudaFree(dev_scanned_indices);
            cudaFree(dev_output);

            return num_elts;
        }
    }
}
