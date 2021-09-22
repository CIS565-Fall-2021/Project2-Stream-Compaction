#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128
#endif // !BLOCKSIZE

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* odata, int d) {

            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            int k = index * (1 << (d + 1));

            odata[k + ((1 << (d + 1)) - 1)] = odata[k + (1 << d) - 1] + odata[k + (1 << (d + 1)) - 1];
        }

        __global__ void kernDownSweep(int n, int* odata, int d) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            int k = index * (1 << (d + 1));
            int t = odata[k + (1 << d) - 1];
            odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];
            odata[k + (1 << (d + 1)) - 1] += t;
        }

        __global__ void kernInvertBoolean(int n, int* bools, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            bools[index] = !idata[index];
        }
        
        __global__ void kernMapBitToBooleans(int n, int bit, int *boolsZero, int *boolsOne, const int *idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
   
            // adapted from stack overflow answer by Sparky: 
            // https://stackoverflow.com/questions/8011700/how-do-i-extract-specific-n-bits-of-a-32-bit-unsigned-integer-in-c 
            int mask = 1 << bit;
		    int isolatedBit = (idata[index] & mask) >> bit;

            boolsOne[index] = isolatedBit;
            boolsZero[index] = !isolatedBit;
        }

        __global__ void kernAddConstant(int n, const int incr, int* odata) {
            // this was simpler than adding a conditional to up/down sweep,
            // had they been implemented differently this could make more sense there
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            odata[index] += incr;
        }

        void scanWithOffset(int n, int* dev_odata, int offset = 0) {
            int numThreads;
            int numBlocks;

            // --- up sweep ---
            for (int d = 0; d < log2(n); d++) {
                numThreads = ((n - 1) / (1 << (d + 1))) + 1;
                numBlocks = ceil((float)numThreads / THREADS_PER_BLOCK);
                kernUpSweep << <numBlocks, THREADS_PER_BLOCK >> > (numThreads, dev_odata, d);
                checkCUDAErrorFn("upsweep failed", "radix.cu", 50);
                //cudaDeviceSynchronize();
                //cudaMemcpy(odata, dev_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // --- down sweep ---
            // insert 0 at the end of the in-progress output
            int ZERO = 0;
            cudaMemcpy(dev_odata + n - 1, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
            for (int d = log2(n - 1); d >= 0; d--) {
                numThreads = ((n - 1) / (1 << (d + 1))) + 1;
                numBlocks = ceil((float)numThreads / THREADS_PER_BLOCK);
                kernDownSweep << <numBlocks, THREADS_PER_BLOCK >> > (numThreads, dev_odata, d);
                checkCUDAErrorFn("downsweep failed", "radix.cu", 65);
                //cudaDeviceSynchronize();
                //cudaMemcpy(odata, dev_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            }

            if (offset) {
                numBlocks = ceil((float)n / THREADS_PER_BLOCK);
                kernAddConstant<<<numBlocks, THREADS_PER_BLOCK>>>(n, offset, dev_odata);
            }
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
        void sort(int n, int* odata, const int* idata) {


            int paddedN = 1 << ilog2ceil(n);
            int* dev_odata1;
            int* dev_odata2;
            int* swp; // for ping-ponging odatas
            int* dev_bitIsOne;
            int* dev_bitIsZero;
            int* dev_indicesZero;
            int* dev_indicesOne;
            
            cudaMalloc((void**)&dev_odata1, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_odata2, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_bitIsOne, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_bitIsZero, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_indicesZero, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_indicesOne, paddedN * sizeof(int));

            cudaMemcpy(dev_odata1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();
            
            // The threads/blocks will change per kernel call but declare them here
			int numBlocks = ceil((float)paddedN / THREADS_PER_BLOCK);
			int numThreads;
            for (int bitNum = 0; bitNum < sizeof(int) * 8; bitNum++) {

                // --- determine radix value ---
                // in this case we check the `bitNum`-th bit and store whether it's a one or zero 
                // in two arrays. why two? so we can scan them both and calculate the proper index
                // for each element based on its current radix
                kernMapBitToBooleans << <numBlocks, THREADS_PER_BLOCK >> > (paddedN, bitNum, dev_bitIsZero, dev_bitIsOne, dev_odata1);
				//cudaDeviceSynchronize();
                //checkCUDAErrorFn("mapToBooleans failed", "radix.cu", 150);

                cudaMemcpy(dev_indicesZero, dev_bitIsZero, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(dev_indicesOne, dev_bitIsOne, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);

                // --- scan ---

                scanWithOffset(paddedN, dev_indicesZero);
                int lastZeroIndexValue;
                int lastZeroIndexBit;
                cudaMemcpy(&lastZeroIndexValue, dev_indicesZero + paddedN - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastZeroIndexBit, dev_bitIsZero + paddedN - 1, sizeof(int), cudaMemcpyDeviceToHost);
				//cudaDeviceSynchronize();
                //checkCUDAErrorFn("memcopy failed", "radix.cu", 155);
                scanWithOffset(paddedN, dev_indicesOne, lastZeroIndexValue + lastZeroIndexBit);
				//cudaDeviceSynchronize();
                //checkCUDAErrorFn("scan failed", "radix.cu", 160);

                // --- scatter ---
                // assign idata -> odata based on the indices calculated by the scan

                Common::kernScatter << <numBlocks, THREADS_PER_BLOCK >> > (paddedN, dev_odata2, dev_odata1, dev_bitIsZero, dev_indicesZero);
				//cudaDeviceSynchronize();
                //checkCUDAErrorFn("scatter failed", "radix.cu", 165);
                Common::kernScatter << <numBlocks, THREADS_PER_BLOCK >> > (paddedN, dev_odata2, dev_odata1, dev_bitIsOne, dev_indicesOne);
				//cudaDeviceSynchronize();
                //checkCUDAErrorFn("scatter failed", "radix.cu", 165);

                // --- ping pong buffers ---
                swp = dev_odata1;
                dev_odata1 = dev_odata2;
                dev_odata2 = swp;
            }

            timer().endGpuTimer();

            // copy the data back to the host
            // offset by (paddedN - n) since any difference between the sizes will
            // yield that many zeros at the beginning of the device array
            cudaMemcpy(odata, dev_odata1 + (paddedN - n), n * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("memcopy failed", "radix.cu", 185);
        }
    }
}
