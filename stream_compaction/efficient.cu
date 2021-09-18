#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 128
int *dev_data_scan;
int* dev_data_scan2;
int *dev_data_compact;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernParallelReduction(int n, int *data, int d, bool lastLoop){
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;

            // start index for each thread
            int startIndex = pow(2, d) - 1;
            // offset the index
            int index = startIndex + pow(2, d+1) * k; 

            int i1 = index;
            int i2 = index + pow(2, d);

            data[i2] += data[i1];
        }

        __global__ void kernDownSweep(int n, int *data, int d){
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            int i1 = index + pow(2, d) - 1;
            int i2 = index + pow(2, d + 1) - 1;

            int leftChild = data[i1];
            data[i1] = data[i2];
            data[i2] += leftChild;
        }

        __global__ void kernMakeExclusiveArray(int n, int* inData, int* outData) {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            // if last index, will be removed
            if (index == n - 1) {
                outData[index] = 0;
                return;
            }
            if (index == 0) {
                outData[0] = 0;
            }
            outData[index+1] = inData[index];
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            // allocate memory
            cudaMalloc((void **)&dev_data_scan, n * sizeof(int));
            cudaMalloc((void **)&dev_data_scan2, n * sizeof(int));
            cudaMemcpy(dev_data_scan, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // perform up-sweep (parallel reduction)
            bool lastLoop = false;
            int numThreads = n;
            dim3 fullBlocksPerGrid;
            for (int d = 0; d < ilog2ceil(n); d++){
                if (d == ilog2ceil(n) - 1) { lastLoop = true; }

                // calc threads and blocks
                numThreads = numThreads / 2.f;
                fullBlocksPerGrid = dim3((numThreads + blockSize - 1) / blockSize);

                kernParallelReduction<<<fullBlocksPerGrid, blockSize>>>(numThreads, dev_data_scan, d, lastLoop);
            }

            // shift array so it's exclusive
            fullBlocksPerGrid = dim3((n + blockSize - 1) / blockSize);
            kernMakeExclusiveArray << <fullBlocksPerGrid, blockSize >> > (n, dev_data_scan, dev_data_scan2);

            // perform down-sweep
            /*for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n, dev_data_scan, d);
            }*/
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data_scan2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data_scan);
            cudaFree(dev_data_scan2);
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
