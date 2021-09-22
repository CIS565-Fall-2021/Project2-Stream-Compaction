#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
#include <cassert>

#define SECTION_SIZE 1024

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void convertFromInclusiveToExclusive(const int* inputArray,
            int* outputArray, int inputSize)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            // convert inclusive scan into exclusive scan by shifting 
            // all elements to the right by one position and fill the frist 
            // element and out-of-bound elements with 0. 
            if (i < inputSize && i != 0)
            {

                outputArray[i] = inputArray[i - 1];
            }
            else {
                outputArray[i] = 0;
            }
        }

        // lanuch this kernel with SECTION_SIZE / 2 threads in a block
        __global__ void kernBrentKungScan(const int* X, int* Y, int* S, int inputSize)
        {
            __shared__ int XY[SECTION_SIZE];
            // 2 * here responsible for handling multiple blocks 
            // now you only consider one block
            int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize)
            {
                XY[threadIdx.x] = X[i];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            if ((i + blockDim.x) < inputSize)
            {
                XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
            }
            else {
                XY[threadIdx.x + blockDim.x] = 0;
            }

            // note here we have stride <= blockDim.x
            for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
            {
                __syncthreads();
                int index = (threadIdx.x + 1) * 2 * stride - 1;
                if (index < SECTION_SIZE)
                {
                    XY[index] += XY[index - stride];
                }
            }

            for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
            {
                __syncthreads();
                int index = (threadIdx.x + 1) * 2 * stride - 1;
                if ((index + stride) < SECTION_SIZE)
                {
                    XY[index + stride] += XY[index];
                }
            }

            __syncthreads();
            if (i < inputSize)
            {
                Y[i] = XY[threadIdx.x];
            }
            else {
                Y[i] = 0;
            }
            if ((i + blockDim.x) < inputSize)
            {
                Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
            }
            else {
                Y[i + blockDim.x] = 0;
            }

            // the last thread in the block should write the output value of 
             // the last XY element in the block to the blockIdx.x position of 
             // SumArray

             // make sure XY[sectionSize - 1] has the correct partial sum
            __syncthreads();
            if (threadIdx.x == blockDim.x - 1)
            {
                S[blockIdx.x] = XY[SECTION_SIZE - 1];
            }
        }

        __global__ void kernBrentKungScan(const int* X, int* Y, int inputSize)
        {
            __shared__ int XY[SECTION_SIZE];
            // 2 * here responsible for handling multiple blocks 
            // now you only consider one block
            int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize)
            {
                XY[threadIdx.x] = X[i];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            if ((i + blockDim.x) < inputSize)
            {
                XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
            }
            else {
                XY[threadIdx.x + blockDim.x] = 0;
            }

            // note here we have stride <= blockDim.x
            for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
            {
                __syncthreads();
                int index = (threadIdx.x + 1) * 2 * stride - 1;
                if (index < SECTION_SIZE)
                {
                    XY[index] += XY[index - stride];
                }
            }

            for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
            {
                __syncthreads();
                int index = (threadIdx.x + 1) * 2 * stride - 1;
                if ((index + stride) < SECTION_SIZE)
                {
                    XY[index + stride] += XY[index];
                }
            }

            __syncthreads();
            if (i < inputSize)
            {
                Y[i] = XY[threadIdx.x];
            }
            else {
                Y[i] = 0;
            }
            if ((i + blockDim.x) < inputSize)
            {
                Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
            }
            else {
                Y[i + blockDim.x] = 0;
            }
        }

        __global__ void kernBrentKungScanAddUpSumArray(const int* S,
            int* Y, int inputSize)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize && blockIdx.x > 0)
            {
                Y[i] += S[blockIdx.x - 1];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // n could be larger than SECTION_SIZE
            int idataSizeBytes = n * sizeof(int);
            int sumArraySizeBytes = (n / SECTION_SIZE) * sizeof(int);

            // MaxThreadsPerBlock: 1024. However, SECTION_SIZE / 2 is needed
            // for kernBrentKungScan
            assert(SECTION_SIZE <= 1024);
            assert(n <= 524288);

            dim3 dimGridBrent((n + (SECTION_SIZE / 2) - 1) / (SECTION_SIZE / 2), 1, 1);
            dim3 dimBlockBrent(SECTION_SIZE / 2, 1, 1);

            dim3 dimGridBrentSumArray(1, 1, 1);
            dim3 dimBlockBrentSumArray(SECTION_SIZE / 2, 1, 1);

            dim3 dimGridArray((n + SECTION_SIZE - 1) / SECTION_SIZE, 1, 1);
            dim3 dimBlockArray(SECTION_SIZE, 1, 1);

            int* d_X;
            int* d_Y;
            int* d_S;
            int* d_SOut;
            int* d_YExclusive;
            cudaMalloc((void**)&d_X, idataSizeBytes);
            checkCUDAError("cudaMalloc d_X failed!");
            cudaMalloc((void**)&d_Y, idataSizeBytes);
            checkCUDAError("cudaMalloc d_Y failed!");
            cudaMalloc((void**)&d_YExclusive, idataSizeBytes);
            checkCUDAError("cudaMalloc d_YExclusive failed!");
            cudaMalloc((void**)&d_S, sumArraySizeBytes);
            checkCUDAError("cudaMalloc d_S failed!");
            cudaMalloc((void**)&d_SOut, sumArraySizeBytes);
            checkCUDAError("cudaMalloc d_SOut failed!");

            cudaMemcpy(d_X, idata, idataSizeBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            kernBrentKungScan << <dimGridBrent, dimBlockBrent >> > (d_X, d_Y, d_S, n);
            kernBrentKungScan << <dimGridBrentSumArray, dimBlockBrentSumArray >> > (d_S, d_SOut, n);
            kernBrentKungScanAddUpSumArray << <dimGridArray, dimBlockArray >> > (d_SOut, d_Y, n);
            convertFromInclusiveToExclusive << <dimGridArray, dimBlockArray >> > (d_Y, d_YExclusive, n);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_YExclusive, idataSizeBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            cudaFree(d_X);
            cudaFree(d_Y);
            cudaFree(d_S);
            cudaFree(d_SOut);
            cudaFree(d_YExclusive);
            checkCUDAError("cudaFree failed!");
        }

        void scanWithoutTimer(int n, int* odata, const int* idata) {
            // n could be larger than SECTION_SIZE
            int idataSizeBytes = n * sizeof(int);
            int sumArraySizeBytes = (n / SECTION_SIZE) * sizeof(int);

            // MaxThreadsPerBlock: 1024. However, SECTION_SIZE / 2 is needed
            // for kernBrentKungScan
            assert(SECTION_SIZE <= 1024);
            assert(n <= 524288);

            dim3 dimGridBrent((n + (SECTION_SIZE / 2) - 1) / (SECTION_SIZE / 2), 1, 1);
            dim3 dimBlockBrent(SECTION_SIZE / 2, 1, 1);

            dim3 dimGridBrentSumArray(1, 1, 1);
            dim3 dimBlockBrentSumArray(SECTION_SIZE / 2, 1, 1);

            dim3 dimGridArray((n + SECTION_SIZE - 1) / SECTION_SIZE, 1, 1);
            dim3 dimBlockArray(SECTION_SIZE, 1, 1);

            int* d_X;
            int* d_Y;
            int* d_S;
            int* d_SOut;
            int* d_YExclusive;
            cudaMalloc((void**)&d_X, idataSizeBytes);
            checkCUDAError("cudaMalloc d_X failed!");
            cudaMalloc((void**)&d_Y, idataSizeBytes);
            checkCUDAError("cudaMalloc d_Y failed!");
            cudaMalloc((void**)&d_YExclusive, idataSizeBytes);
            checkCUDAError("cudaMalloc d_YExclusive failed!");
            cudaMalloc((void**)&d_S, sumArraySizeBytes);
            checkCUDAError("cudaMalloc d_S failed!");
            cudaMalloc((void**)&d_SOut, sumArraySizeBytes);
            checkCUDAError("cudaMalloc d_SOut failed!");

            cudaMemcpy(d_X, idata, idataSizeBytes, cudaMemcpyHostToDevice);

            kernBrentKungScan << <dimGridBrent, dimBlockBrent >> > (d_X, d_Y, d_S, n);
            kernBrentKungScan << <dimGridBrentSumArray, dimBlockBrentSumArray >> > (d_S, d_SOut, n);
            kernBrentKungScanAddUpSumArray << <dimGridArray, dimBlockArray >> > (d_SOut, d_Y, n);
            convertFromInclusiveToExclusive << <dimGridArray, dimBlockArray >> > (d_Y, d_YExclusive, n);

            cudaMemcpy(odata, d_YExclusive, idataSizeBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            cudaFree(d_X);
            cudaFree(d_Y);
            cudaFree(d_S);
            cudaFree(d_SOut);
            cudaFree(d_YExclusive);
            checkCUDAError("cudaFree failed!");
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
            timer().startCpuTimer();

            int numElement = 0;
            std::unique_ptr<int[]>tempArray{ new int[n] };
            std::unique_ptr<int[]>scanResult{ new int[n] };
            for (int i = 0; i < n; i++)
            {
                scanResult[i] = -1;
            }

            // STEP 1: Compute temp Array with 0s and 1s
            // intialize array such that all elements meet criteria
            for (int i = 0; i < n; i++)
            {
                tempArray[i] = 1;
            }
            // next, figure out which one doesn't meet criteria
            for (int i = 0; i < n; i++)
            {
                // since we want to remove 0s, elements with value = 0 doesn't
                // meet criteria
                if (idata[i] == 0)
                {
                    tempArray[i] = 0;
                }
            }

            // STEP 2: Run exclusive scan on tempArray
            scanWithoutTimer(n, scanResult.get(), tempArray.get());

            // STEP 3: scatter
            for (int i = 0; i < n; i++)
            {
                // result of scan is index into final array
                int index = scanResult[i];
                // only write an element if temp array has a 1
                if (tempArray[i] == 1)
                {
                    odata[index] = idata[i];
                    numElement++;
                }
            }

            timer().endCpuTimer();
            return n - numElement;
        }
    }
}
