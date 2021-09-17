#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
dim3 threadsPerBlock(blockSize);

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

        __global__ void kernParallelReduction(int n, int *data, int d){
            //int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            //data[index + pow(2, d+1) - 1] += data[index + pow(2, d) - 1];
        }

        __global__ void kernDownSweep(int n, int *data, int d){
            //int index = (blockDim.x + blockIdx.x) + threadIdx.x;
            //int leftChild = data[index + pow(2, d) - 1];
           // data[index + pow(2, d) - 1] = data[index + pow(2, d+1) - 1];
           // data[index + pow(2, d+1) - 1] += leftChild;
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            // allocate memory
            /*dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            cudaMalloc((void **)&dev_data_scan, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_data failed!");

            timer().startGpuTimer();

            // perform up-sweep (parallel reduction)
            for (int d = 0; d < ilog2ceil(n) - 1; d++){
                kernParallelReduction<<<fullBlocksPerGrid, blockSize>>>(n, dev_data_scan, d);
            }

            // perform down-sweep
            dev_data_scan[n - 1] = 0;
            for (int d = ilog2ceil(n) - 1; d >= 0; d--){
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n, dev_data_scan, d);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data_scan, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data_scan);*/
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
