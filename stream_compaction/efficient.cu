#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweepIter(int nPadded, int depth, int* dataPadded) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return; 
            }

            dataPadded[index + (1 << (depth + 1)) - 1] += index % (1 << (depth + 1)) == 0 ? dataPadded[index + (1 << depth) - 1] : 0; 
        }

        __global__ void kernDownSweepIter(int nPadded, int depth, int* dataPadded) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return;
            }

            int offsetPlus = 1 << (depth + 1); 
            if (index % offsetPlus == 0) {
                int temp = dataPadded[index + (offsetPlus >> 1) - 1];
                dataPadded[index + (offsetPlus >> 1) - 1] = dataPadded[index + offsetPlus - 1];
                dataPadded[index + offsetPlus - 1] += temp;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            if (n < 1) { return; }

            // allocate a buffer padded to a power of 2. 
            int depth = ilog2ceil(n);
            int nPadded = 1 << depth;

            int* dev_dataPadded;
            cudaMalloc((void**)&dev_dataPadded, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataPadded failed!");

            // set blocks and threads 
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid(std::ceil((double)nPadded / blockSize));

            // copy idata to device memory 
            cudaMemset(dev_dataPadded, 0, nPadded * sizeof(int));
            cudaMemcpy(dev_dataPadded, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            // perform upsweep on idata
            for (int i = 0; i < depth; i++) {
                kernUpSweepIter<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_dataPadded); 
            }

            // perform downsweep on idata
            cudaMemset(dev_dataPadded + nPadded - 1, 0, sizeof(int)); 
            for (int i = depth - 1; i >= 0; i--) {
                kernDownSweepIter<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_dataPadded); 
            }

            cudaDeviceSynchronize();
            timer().endGpuTimer();

            // copy scan back to host
            cudaMemcpy(odata, dev_dataPadded, n * sizeof(int), cudaMemcpyDeviceToHost);
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
