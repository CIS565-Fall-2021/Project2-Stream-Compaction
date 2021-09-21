#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // DONE-Part 1

        /**
         * Inserts a 0 (identity) into the beginning of the scanned array
         */
        __global__ void kernInsertZero(int N, int* data1, int* data2) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N)
            {
                return;
            }

            if (index == 0)
            {
                data2[index] = 0;
            }
            else
            {
                data2[index] = data1[index - 1];
            }
        }

        /**
         * Performs an inclusive scan on a set of data
         */
        __global__ void kernScanNaive(int N, int d, int* data1, int* data2) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N)
            {
                return;
            }

            if (index >= (1 << (d - 1)))
            {
                data2[index] = data1[index - (1 << (d - 1))] + data1[index];
            }
            else
            {
                data2[index] = data1[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int N, int *odata, const int *idata) {
            // CUDA array initialization
            unsigned int num_bytes = N * sizeof(int);
            int* dev_data1;
            int* dev_data2;
            cudaMalloc((void**)&dev_data1, num_bytes);
            cudaMalloc((void**)&dev_data2, num_bytes);
            cudaMemcpy(dev_data1, idata, num_bytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            unsigned int max_depth = ilog2ceil(N);
            for (int d = 1; d <= max_depth; d++)
            {
                // Perform inclusive scan
                kernScanNaive<<<fullBlocksPerGrid, BLOCK_SIZE>>>(N, d, dev_data1, dev_data2);

                int *temp = dev_data2;
                dev_data2 = dev_data1;
                dev_data1 = temp;
            }

            // Convert from inclusive to exclusive scan
            kernInsertZero<<<fullBlocksPerGrid, BLOCK_SIZE>>>(N, dev_data1, dev_data2);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data2, num_bytes, cudaMemcpyDeviceToHost);
            cudaFree(dev_data1);
            cudaFree(dev_data2);

        }
    }
}
