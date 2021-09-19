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
            int* dev_odata; 

            // pad to a power of 2
            int paddedN = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_odata, paddedN * sizeof(int));

            // write n items to the GPU array, the total length is `paddedN`, meaning arr[n:paddedN] are 0 
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // --- up sweep ---
            for (int d = 0; d < log2(paddedN); d++) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
				kernUpSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_odata, d);
				checkCUDAErrorFn("upsweep failed", "efficent.cu", 50);
				cudaDeviceSynchronize();
				//cudaMemcpy(odata, dev_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // --- down sweep ---
            // insert 0 at the end of the in-progress output
            int ZERO = 0;
            cudaMemcpy(dev_odata + paddedN - 1, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
            for (int d = log2(paddedN - 1); d >= 0; d--) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
				kernDownSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_odata, d);
				checkCUDAErrorFn("downsweep failed", "efficent.cu", 70);
				cudaDeviceSynchronize();
				//cudaMemcpy(odata, dev_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            timer().endGpuTimer();

            // this is an exclusive scan, so the first elem should be 0
            // and we shift everything (except the last elem) one index right
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
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

            int paddedN = 1<<ilog2ceil(n);
            int* dev_odata;
            int* dev_idata;
            int* dev_hasElem;
            int* dev_indices;

            cudaMalloc((void**)&dev_odata, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_idata, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_hasElem, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            Common::kernMapToBoolean<<<n, BLOCKSIZE >>>(n, dev_hasElem, dev_idata);

            cudaMemcpy(dev_indices, dev_hasElem, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // --- scan ---
            // scanning the `hasElem` "boolean" array yields an array of indices
            // idata[i] should be assigned to (if hasElem[i] is truthy).
            // why copy (most of the) code from scan() instead of calling it?
            // to avoid redundant cudaMalloc/cudaMemcpy calls

            // --- up sweep ---
            for (int d = 0; d < log2(paddedN); d++) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
				kernUpSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_indices, d);
				checkCUDAErrorFn("upsweep failed", "efficent.cu", 50);
				cudaDeviceSynchronize();
				//cudaMemcpy(odata, dev_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // --- down sweep ---
            // insert 0 at the end of the in-progress output
            int ZERO = 0;
            cudaMemcpy(dev_indices + paddedN - 1, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
            for (int d = log2(paddedN - 1); d >= 0; d--) {
				int numThreads = ((paddedN - 1) / (1 << (d + 1))) + 1;
				kernDownSweep <<<numThreads, BLOCKSIZE>>> (numThreads, dev_indices, d);
				checkCUDAErrorFn("downsweep failed", "efficent.cu", 70);
				cudaDeviceSynchronize();
				//cudaMemcpy(odata, dev_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // --- scatter ---
            // assign idata -> odata based on the indices calculated by the scan

            Common::kernScatter<<<n, BLOCKSIZE>>>(n, dev_odata, dev_idata, dev_hasElem, dev_indices);

            timer().endGpuTimer();

            // get the max index for return purposes. This will be whatever is at the end
            // of our scattered index array
            int maxIndex;
            cudaMemcpy(&maxIndex, dev_indices + paddedN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(odata, dev_odata, maxIndex * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);

            return maxIndex;
        }
    }
}
