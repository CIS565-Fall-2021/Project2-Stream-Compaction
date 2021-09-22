#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 256
int *dev_data_scan;
int *dev_compact_in;
int* dev_compact_out;
int* dev_bools;
int* dev_indices;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernParallelReduction(int *data, int d, int powd, int powd1){
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;

            // offset the index
            int index = (powd - 1) + powd1 * k;

            int i1 = index;
            int i2 = index + powd;

            data[i2] += data[i1];
        }

        __global__ void kernDownSweep(int *data, int d, bool firstLoop, int powd, int powd1){
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            int offset = index * (powd1 - 1);
            int i1 = index + powd - 1 + offset;
            int i2 = index + powd1 - 1 + offset;

            // set last item to 0 on first loop
            if (firstLoop) {
                data[i2] = 0;
            }
            int leftChild = data[i1];
            data[i1] = data[i2];
            data[i2] += leftChild;
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            int roundedN = pow(2, ilog2ceil(n));
            // allocate memory
            cudaMalloc((void **)&dev_data_scan, roundedN * sizeof(int));
            cudaMemcpy(dev_data_scan, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // perform up-sweep (parallel reduction)
            int numThreads = roundedN;
            dim3 fullBlocksPerGrid;
            for (int d = 0; d < ilog2ceil(n); d++){
                // calc threads and blocks
                numThreads = numThreads / 2;
                fullBlocksPerGrid = dim3((numThreads + blockSize - 1) / blockSize);

                int powd = pow(2, d);
                int powd1 = pow(2, d + 1);
                kernParallelReduction<<<fullBlocksPerGrid, numThreads>>>(dev_data_scan, d, powd, powd1);
            }

            // perform down-sweep
            bool firstLoop = false;
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                fullBlocksPerGrid = dim3((numThreads + blockSize - 1) / blockSize);
                firstLoop = d == ilog2ceil(n) - 1;

                int powd = pow(2, d);
                int powd1 = pow(2, d + 1);
                kernDownSweep<<<fullBlocksPerGrid, numThreads>>>(dev_data_scan, d, firstLoop, powd, powd1);

                numThreads = numThreads * 2;
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data_scan, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data_scan);
        }

        /**
        * Performs prefix-sum (aka scan) on odata, storing the result into odata. 
        * Both ptrs are device ptrs. This is a helper for compact()
        */
        int scanHelper(int n, int* odata)
        {
            int roundedN = pow(2, ilog2ceil(n));
            // allocate memory
            cudaMalloc((void**)&dev_data_scan, roundedN * sizeof(int));
            cudaMemcpy(dev_data_scan, odata, n * sizeof(int), cudaMemcpyHostToDevice);

            // perform up-sweep (parallel reduction)
            int numThreads = roundedN;
            dim3 fullBlocksPerGrid;
            for (int d = 0; d < ilog2ceil(n); d++) {
                // calc threads and blocks
                numThreads = numThreads / 2;
                fullBlocksPerGrid = dim3((numThreads + blockSize - 1) / blockSize);

                int powd = pow(2, d);
                int powd1 = pow(2, d + 1);
                kernParallelReduction << <fullBlocksPerGrid, numThreads >> > (dev_data_scan, d, powd, powd1);
            }
            // save size of compact array from last elem of reduction
            int compactSize;
            cudaMemcpy(&compactSize, dev_data_scan + roundedN - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // perform down-sweep
            bool firstLoop = false;
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                fullBlocksPerGrid = dim3((numThreads + blockSize - 1) / blockSize);
                firstLoop = d == ilog2ceil(n) - 1;

                int powd = pow(2, d);
                int powd1 = pow(2, d + 1);
                kernDownSweep << <fullBlocksPerGrid, numThreads >> > (dev_data_scan, d, firstLoop, powd, powd1);

                numThreads = numThreads * 2;
            }

            cudaMemcpy(odata, dev_data_scan, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data_scan);
            return compactSize;
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
            // allocate memory
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            cudaMalloc((void **)&dev_compact_in, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMemcpy(dev_compact_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // map bools to ints
            Common::kernMapToBoolean<<<fullBlocksPerGrid, n>>>(n, dev_bools, dev_compact_in);

            cudaMemcpy(odata, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

            int outSize = scanHelper(n, odata);
            cudaMalloc((void**)&dev_compact_out, outSize * sizeof(int));

            // perform scatter to fill final array
            Common::kernScatter<<<fullBlocksPerGrid, n>>>(n, dev_compact_out, dev_compact_in, 
                                                          dev_bools, dev_data_scan);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_compact_out, outSize * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_compact_in);
            cudaFree(dev_bools);
            cudaFree(dev_compact_out);
            cudaFree(dev_indices);

            return outSize;
        }
    }
}
