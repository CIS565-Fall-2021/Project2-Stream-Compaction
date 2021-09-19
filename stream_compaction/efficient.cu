#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE 128
#endif // !BLOCKSIZE

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernUpSweep(int n, int *odata, int d) {

            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index > n) {
                return;
            }

            int k = index * (1 << (d + 1));

			odata[k + ((1<<(d+1))-1)] = odata[k + (1 << d) - 1] + odata[k + (1 << (d+1)) - 1];
            __syncthreads();
        }

        __global__ void kernDownSweep(int n, int *odata, int d) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index > n) {
                return;
            }

            int k = index * (1 << (d + 1));
            int t = odata[k + (1 << d) - 1];
            odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];
            odata[k + (1 << (d + 1)) - 1] += t;
            __syncthreads();
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
           
            int* dev_readable; 
            int* dev_writeable; 

            // pad to a power of 2
            int paddedN = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_writeable, paddedN * sizeof(int));

            // write n items to the GPU array, but offset the start so that the total length is `paddedN` 
            cudaMemcpy(dev_writeable, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // --- up sweep ---
            for (int d = 0; d < log2(paddedN); d++) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
				kernUpSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_writeable, d);
				checkCUDAErrorFn("upsweep failed", "efficent.cu", 50);
				cudaDeviceSynchronize();
				//cudaMemcpy(odata, dev_writeable, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // --- down sweep ---
            // insert 0 at the end of the in-progress output
            int ZERO = 0;
            cudaMemcpy(dev_writeable + paddedN - 1, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(odata, dev_writeable, n * sizeof(int), cudaMemcpyDeviceToHost);
            for (int d = log2(paddedN - 1); d >= 0; d--) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
				kernDownSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_writeable, d);
				checkCUDAErrorFn("downsweep failed", "efficent.cu", 70);
				cudaDeviceSynchronize();
				//cudaMemcpy(odata, dev_writeable, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            timer().endGpuTimer();

            // this is an exclusive scan, so the first elem should be 0
            // and we shift everything (except the last elem) one index right
            cudaMemcpy(odata, dev_writeable, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_writeable);
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
