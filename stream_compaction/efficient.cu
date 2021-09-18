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
            int k = index * (1 << (d + 1));

            if (index > n) {
                return;
            }

			odata[k + ((1<<(d+1))-1)] = odata[k + (1 << d) - 1] + odata[k + (1 << (d+1)) - 1];
            __syncthreads();
            return;
        }

        __global__ void kernDownSweep(int n, int *odata, const int *idata, int d) {

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
           
            int* dev_readable; 
            int* dev_writeable; 

            int paddedN = 1 << ilog2ceil(n);

            //cudaMalloc((void**)&dev_readable, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_writeable, paddedN * sizeof(int));

            cudaMemcpy(dev_writeable, idata, paddedN * sizeof(int), cudaMemcpyHostToDevice);

            // --- up sweep ---
            for (int d = 0; d < log2(paddedN); d++) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
                //numthreads = numThreads > 0 ? numthreads : 1;
                printf("blocksize: %i\t numthreads: %i\n", BLOCKSIZE, numThreads);
				kernUpSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_writeable, d);
				checkCUDAErrorFn("upsweep failed", "efficent.cu", 50);
				cudaDeviceSynchronize();
				cudaMemcpy(odata, dev_writeable, n * sizeof(int), cudaMemcpyDeviceToHost);
            }


            // --- down sweep ---
            // insert 0 at the end of the in-progress output
            //cudaMemcpy(dev_writeable + n - 1, { 0 }, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(odata, dev_writeable, n * sizeof(int), cudaMemcpyDeviceToHost);
            for (int d = 0; d < log2(paddedN); d++) {
				//int numThreads = (paddedN - 1) / (1 << d + 1);
				//kernDownSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_writeable, dev_readable, d);
				//checkCUDAErrorFn("downSweep failed", "efficent.cu", 70);
				//cudaDeviceSynchronize();
            }

            timer().endGpuTimer();

            // this is an exclusive scan, so the first elem should be 0
            // and we shift everything (except the last elem) one index right
            odata[0] = 0; // TODO try removing this
            cudaMemcpy(odata+1, dev_writeable, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_readable);
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
