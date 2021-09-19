#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 128
int *dev_data_scan;
int *dev_data_compact;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernParallelReduction(int *data, int d){
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;

            // start index for each thread
            int startIndex = pow(2, d) - 1;
            // offset the index
            int index = startIndex + pow(2, d+1) * k; 

            int i1 = index;
            int i2 = index + pow(2, d);

            data[i2] += data[i1];
        }

        __global__ void kernDownSweep(int *data, int d, bool firstLoop){
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            int offset = index * (pow(2, d + 1) - 1);
            int i1 = index + pow(2, d) - 1 + offset;
            int i2 = index + pow(2, d + 1) - 1 + offset;

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

                kernParallelReduction<<<fullBlocksPerGrid, numThreads>>>(dev_data_scan, d);
            }

            // perform down-sweep
            bool firstLoop = false;
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                fullBlocksPerGrid = dim3((numThreads + blockSize - 1) / blockSize);
                firstLoop = d == ilog2ceil(n) - 1;

                kernDownSweep<<<fullBlocksPerGrid, numThreads>>>(dev_data_scan, d, firstLoop);

                numThreads = numThreads * 2;
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data_scan, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data_scan);
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
            /*dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            cudaMalloc((void **)&dev_data_compact, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_data failed!");

            timer().startGpuTimer();
            // map bools to ints
            kernMapToBoolean<<<fullBlocksPerGrid, threadsPerBlock>>>(n, idata, dev_data_compact);

            scan(n, dev_data_compact, dev_data_compact);

            // TODO: revisit once have access to lab computer
            // change way you handle dev_data
            kernScatter<<<fullBlocksPerGrid, threadsPerBlock>>>(n, odata, idata, dev_data_compact, dev_data_compact);

            timer().endGpuTimer();

            cudaFree(dev_data_compact);

            return dev_data_compact[n-1];*/
            return 0;
        }
    }
}
