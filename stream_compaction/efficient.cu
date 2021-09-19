#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernIncrement(int n, int* global_odata, int* global_blockIncrements) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;

          if (index < n) {
            global_odata[2 * index + 0] += global_blockIncrements[blockIdx.x];
            global_odata[2 * index + 1] += global_blockIncrements[blockIdx.x];
          }
        }

        __global__ void kernScan(int nThreadsNeeded, int n, int* global_idata, int* global_odata, int* global_blockSums) {
          // n is 2x the blockSize (total num elements being operated on by the block)
          extern __shared__ int shared_array[];

          int index = (blockIdx.x * blockDim.x) + threadIdx.x;

          if (index >= nThreadsNeeded) {
            return;
          }

          shared_array[2 * threadIdx.x + 0] = global_idata[2 * index + 0];
          shared_array[2 * threadIdx.x + 1] = global_idata[2 * index + 1];

          // SWEEP UP
          int offset = 1;
          // Cute way of looping e.g. if n = 1024, then d = 512, 256, 128, 64, 32, 16, 8, 4, 2, 1
          for (int d = n >> 1; d > 0; d >>= 1) {
            __syncthreads();
            
            if (threadIdx.x < d) {
              int thisThreadsIndex1 = offset * (2 * threadIdx.x + 1) - 1;
              int thisThreadsIndex2 = offset * (2 * threadIdx.x + 2) - 1;

              shared_array[thisThreadsIndex2] += shared_array[thisThreadsIndex1];
            }

            offset *= 2;
          }

          // SWEEP DOWN
          if (threadIdx.x == 0) {
            if (global_blockSums) {
              global_blockSums[blockIdx.x] = shared_array[n - 1];
            }

            shared_array[n - 1] = 0;
          }

          for (int d = 1; d < n; d *= 2) {
            // offset begins at 1024 and is set to 512
            offset >>= 1;
            __syncthreads();

            if (threadIdx.x < d) {
              int thisThreadsIndex1 = offset * (2 * threadIdx.x + 1) - 1;
              int thisThreadsIndex2 = offset * (2 * threadIdx.x + 2) - 1;

              float tmp = shared_array[thisThreadsIndex1];
              shared_array[thisThreadsIndex1] = shared_array[thisThreadsIndex2];
              shared_array[thisThreadsIndex2] += tmp;
            }
          }

          __syncthreads();
          global_odata[2 * index + 0] = shared_array[2 * threadIdx.x + 0];
          global_odata[2 * index + 1] = shared_array[2 * threadIdx.x + 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        int* scan(int n, int *odata, int *idata) {
          int paddedN = int(pow(2.0, ilog2ceil(n)));
          int nThreadsNeeded = paddedN / 2;
          int blockSize = std::min(256, nThreadsNeeded);
          int nBlocks = ceil(nThreadsNeeded * 1.0 / blockSize);
          int nElementsPerBlock = 2 * blockSize;

          std::cout << std::endl;
          std::cout << "paddedN: " << paddedN << std::endl;
          std::cout << "grid size: " << nBlocks << std::endl;
          std::cout << "block size: " << blockSize << std::endl;
          

          int* device_odata;
          cudaMalloc((void**)&device_odata, paddedN * sizeof(int));
          checkCUDAError("cudaMalloc device_odata failed!");
          cudaDeviceSynchronize();

          if (nBlocks == 1) {
            if (odata) {
              int* device_idata;
              cudaMalloc((void**)&device_idata, paddedN * sizeof(int));
              checkCUDAError("cudaMalloc device_idata failed!");
              cudaDeviceSynchronize();

              cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
              checkCUDAError("cudaMalloc device_idata failed!");
              cudaDeviceSynchronize();

              timer().startGpuTimer();
              kernScan << <nBlocks, blockSize, nElementsPerBlock * sizeof(int) >> > (nThreadsNeeded, nElementsPerBlock, device_idata, device_odata, NULL);
              timer().endGpuTimer();
              checkCUDAError("kernScan failed!");
              cudaDeviceSynchronize();

              cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
              checkCUDAError("cudaMemcpy device_odata failed!");
              cudaDeviceSynchronize();
              return NULL;
            } else {
              kernScan << <1, blockSize, nElementsPerBlock * sizeof(int) >> > (nThreadsNeeded, nElementsPerBlock, idata, device_odata, NULL);
              checkCUDAError("kernScan failed!");
              cudaDeviceSynchronize();

              return device_odata;
            }
          } else {
            int* device_idata;
            cudaMalloc((void**)&device_idata, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc device_idata failed!");
            cudaDeviceSynchronize();

            int* device_blockSums;
            cudaMalloc((void**)&device_blockSums, nBlocks * sizeof(int));
            checkCUDAError("cudaMalloc device_blockSums failed!");
            cudaDeviceSynchronize();

            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");
            cudaDeviceSynchronize();

            //if (odata) {
            //  timer().startGpuTimer();
            //}
            kernScan << <nBlocks, blockSize, nElementsPerBlock * sizeof(int) >> > (nThreadsNeeded, nElementsPerBlock, device_idata, device_odata, device_blockSums);
            checkCUDAError("kernScan failed!");
            cudaDeviceSynchronize();

            int* device_blockIncrements = scan(nBlocks, NULL, device_blockSums); // (blockSumScan)
            checkCUDAError("Recursive call failed!");
            cudaDeviceSynchronize();
            

            kernIncrement << <nBlocks, blockSize >> > (nThreadsNeeded, device_odata, device_blockIncrements);
            checkCUDAError("kernIncrement failed!");
            cudaDeviceSynchronize();
 /*           if (odata) {
              timer().endGpuTimer();
            }*/

            if (!odata) {
              return device_odata;
            }
          }

          cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy failed!");
          cudaDeviceSynchronize();

          return NULL;
        }

        __global__ void kernMapToBoolean(int n, int* global_idata, int* global_booleanMask) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;

          if (n <= index) {
            return;
          }

          global_booleanMask[index] = global_idata[index] ? 1 : 0;
        }

        __global__ void kernScatter(int n, int* global_idata, int* global_odata, int* global_booleanMask, int* global_booleanMaskScan) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;

          if (n <= index) {
            return;
          }

          if (global_booleanMask[index]) {
            global_odata[global_booleanMaskScan[index]] = global_idata[index];
          }

          __syncthreads();
          if (index == 0) {
            global_booleanMaskScan[n - 1] += global_booleanMask[n - 1];
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
        int compact(int n, int *odata, int *idata) {
          int paddedN = int(pow(2, ilog2ceil(n)));
          int blockSize = 256;
          int nBlocks = ceil(paddedN * 1.0 / blockSize);

          int* device_idata;
          cudaMalloc((void**)&device_idata, paddedN * sizeof(int));
          checkCUDAError("cudaMalloc device_idata failed!");

          int* device_booleanMask;
          cudaMalloc((void**)&device_booleanMask, paddedN * sizeof(int));
          checkCUDAError("cudaMalloc device_booleanMask failed!");

          int* device_odata;
          cudaMalloc((void**)&device_odata, paddedN * sizeof(int));
          checkCUDAError("cudaMalloc device_odata failed!");

          cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          cudaDeviceSynchronize();

          timer().startGpuTimer();
          kernMapToBoolean << <nBlocks, blockSize >> > (n, device_idata, device_booleanMask);
          cudaDeviceSynchronize();
          timer().endGpuTimer();

          int* device_booleanMaskScan = scan(n, NULL, device_booleanMask);

          kernScatter << <nBlocks, blockSize >> > (n, device_idata, device_odata, device_booleanMask, device_booleanMaskScan);
          

          int size;
          cudaMemcpy(&size, device_booleanMaskScan + paddedN - 1, sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(odata, device_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);

          return size;
        }
    }
}
