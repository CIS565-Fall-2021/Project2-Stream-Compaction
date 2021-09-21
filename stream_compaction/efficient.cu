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

        __device__ void reductionStep(int *XY)
        {
            for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
            {
                // make sure that input is in place
                __syncthreads();
                int index = (threadIdx.x + 1) * stride * 2 - 1;
                if (index < sectionSize)
                {
                    XY[index] += XY[index - stride];
                }
            }
        }

        __device__ void postScanStep(int* XY)
        {
            for (unsigned int stride = sectionSize / 4; stride > 0; stride /= 2)
            {
                // make sure that input is in place
                __syncthreads();
                int index = (threadIdx.x + 1) * stride * 2 - 1;
                if ((index + stride) < sectionSize)
                {
                    XY[index + stride] += XY[index];
                }
            }
        }

        __device__ void computeScanToOutputArray(const int* inputArray, int* outputArray,
            int* XY, int inputSize)
        {
            int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
            // each thread loads two input elements into the shared memory
            if (i < inputSize)
            {
                XY[threadIdx.x] = inputArray[i];
            }
            if (i + blockDim.x < inputSize)
            {
                XY[threadIdx.x + blockDim.x] = inputArray[i + blockDim.x];
            }
            reductionStep(XY);
            postScanStep(XY);
            // each thread write two elements into the output array
            __syncthreads();
            if (i < inputSize)
            {
                outputArray[i] = XY[threadIdx.x];
            }
            if (i + blockDim.x < inputSize)
            {
                outputArray[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
            }
        }

        __global__ void kernWorkEfficientGPUScanFirstStep(const int* inputArray,
            int* outputArray, int* SumArray, int inputSize)
        {
            __shared__ int XY[sectionSize];
            computeScanToOutputArray(inputArray, outputArray, XY, inputSize);

            // the last thread in the block should write the output value of 
            // the last XY element in the block to the blockIdx.x position of 
            // SumArray

            // make sure XY[sectionSize - 1] has the correct partial sum
            __syncthreads();
            if (threadIdx.x == blockDim.x - 1)
            {
                SumArray[blockIdx.x] = XY[sectionSize - 1];
            }
        }

        __global__ void kernWorkEfficientGPUScanSecondStep(const int* inputArray,
            int* outputArray, int inputSize)
        {
            __shared__ int XY[sectionSize];
            computeScanToOutputArray(inputArray, outputArray, XY, inputSize);
        }


        __global__ void kernWorkEfficientGPUScanThirdStep(const int* inputArray,
            int* outputArray, int inputSize)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize && blockIdx.x > 0)
            {
                outputArray[i] += inputArray[blockIdx.x - 1];
            }
        }

        void scanWithoutTimer(int n, int* odata, const int* idata) {
            int size = n * sizeof(int);
            int sumArrayNumEle = (n + blockSize - 1) / blockSize;
            int sumArraySize = sumArrayNumEle * sizeof(int);

            int* d_InputData;
            int* d_OutputData;
            int* d_OutputExclusiveData;
            int* d_SumArray;
            int* d_SumArrayOutput;

            cudaMalloc((void**)&d_InputData, size);
            checkCUDAError("cudaMalloc d_InputData failed!");

            cudaMalloc((void**)&d_OutputData, size);
            checkCUDAError("cudaMalloc d_OutputData failed!");

            cudaMalloc((void**)&d_OutputExclusiveData, size);
            checkCUDAError("cudaMalloc d_OutputExclusiveData failed!");

            cudaMalloc((void**)&d_SumArray, sumArraySize);
            checkCUDAError("cudaMalloc d_SumArray failed!");

            cudaMalloc((void**)&d_SumArrayOutput, sumArraySize);
            checkCUDAError("cudaMalloc d_SumArrayOutput failed!");

            cudaMemcpy(d_InputData, idata, size, cudaMemcpyHostToDevice);

            // Only need to launch a kernel with (blockSize / 2) in a block
            // b/c each thread loads/stores two elements
            dim3 dimGridArrayEfficient((n + (blockSize / 2) - 1) / (blockSize / 2), 1, 1);
            dim3 dimBlockArrayEfficient((blockSize / 2), 1, 1);

            dim3 dimGridSumArray((sumArrayNumEle + (blockSize / 2) - 1) / (blockSize / 2), 1, 1);
            dim3 dimBlockSumArray((blockSize / 2), 1, 1);

            dim3 dimGridArray((n + blockSize - 1) / blockSize, 1, 1);
            dim3 dimBlockArray(blockSize, 1, 1);

            // timer().startGpuTimer();

            // First step: compute the scan result for individual sections
            // then, store their block sum to sumArray
            kernWorkEfficientGPUScanFirstStep << <dimGridArrayEfficient,
                dimBlockArrayEfficient >> > (d_InputData, d_OutputData,
                    d_SumArray, n);
            checkCUDAError("kernNaiveGPUScanFirstStep failed!");

            // cudaDeviceSynchronize();

            // Second step: scan block sums
            kernWorkEfficientGPUScanSecondStep << <dimGridSumArray, dimBlockSumArray >> > (
                d_SumArray, d_SumArrayOutput, sumArrayNumEle);
            checkCUDAError("kernNaiveGPUScanSecondStep failed!");

            // cudaDeviceSynchronize();

            // Third step: add scanned block sum i to all values of scanned block
            // i + 1
            kernWorkEfficientGPUScanThirdStep << <dimGridArray, dimBlockArray >> > (
                d_SumArrayOutput, d_OutputData, n);
            checkCUDAError("kernNaiveGPUScanThirdStep failed!");

            // cudaDeviceSynchronize();

            // Last step:
            convertFromInclusiveToExclusive << <dimGridArray, dimBlockArray >> > (
                d_OutputData, d_OutputExclusiveData, n);
            checkCUDAError("convertFromInclusiveToExclusive failed!");
            // timer().endGpuTimer();

            cudaMemcpy(odata, d_OutputExclusiveData, size, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            // cleanup
            cudaFree(d_InputData);
            cudaFree(d_OutputData);
            cudaFree(d_OutputExclusiveData);
            cudaFree(d_SumArray);
            cudaFree(d_SumArrayOutput);
            checkCUDAError("cudaFree failed!");
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = n * sizeof(int);
            int sumArrayNumEle = (n + blockSize - 1) / blockSize;
            int sumArraySize = sumArrayNumEle * sizeof(int);

            int* d_InputData;
            int* d_OutputData;
            int* d_OutputExclusiveData;
            int* d_SumArray;
            int* d_SumArrayOutput;

            cudaMalloc((void**)&d_InputData, size);
            checkCUDAError("cudaMalloc d_InputData failed!");

            cudaMalloc((void**)&d_OutputData, size);
            checkCUDAError("cudaMalloc d_OutputData failed!");

            cudaMalloc((void**)&d_OutputExclusiveData, size);
            checkCUDAError("cudaMalloc d_OutputExclusiveData failed!");

            cudaMalloc((void**)&d_SumArray, sumArraySize);
            checkCUDAError("cudaMalloc d_SumArray failed!");

            cudaMalloc((void**)&d_SumArrayOutput, sumArraySize);
            checkCUDAError("cudaMalloc d_SumArrayOutput failed!");

            cudaMemcpy(d_InputData, idata, size, cudaMemcpyHostToDevice);

            // Only need to launch a kernel with (blockSize / 2) in a block
            // b/c each thread loads/stores two elements
            dim3 dimGridArrayEfficient((n + (blockSize / 2) - 1) / (blockSize / 2), 1, 1);
            dim3 dimBlockArrayEfficient((blockSize / 2), 1, 1);

            dim3 dimGridSumArray((sumArrayNumEle + (blockSize / 2) - 1) / (blockSize / 2), 1, 1);
            dim3 dimBlockSumArray((blockSize / 2), 1, 1);

            dim3 dimGridArray((n + blockSize - 1) / blockSize, 1, 1);
            dim3 dimBlockArray(blockSize, 1, 1);

            timer().startGpuTimer();

            // First step: compute the scan result for individual sections
            // then, store their block sum to sumArray
            kernWorkEfficientGPUScanFirstStep <<<dimGridArrayEfficient, 
                dimBlockArrayEfficient >> > (d_InputData, d_OutputData, 
                    d_SumArray, n);
            checkCUDAError("kernNaiveGPUScanFirstStep failed!");

            // cudaDeviceSynchronize();

            // Second step: scan block sums
            kernWorkEfficientGPUScanSecondStep << <dimGridSumArray, dimBlockSumArray >>> (
                d_SumArray, d_SumArrayOutput, sumArrayNumEle);
            checkCUDAError("kernNaiveGPUScanSecondStep failed!");

            // cudaDeviceSynchronize();

            // Third step: add scanned block sum i to all values of scanned block
            // i + 1
            kernWorkEfficientGPUScanThirdStep << <dimGridArray, dimBlockArray >>> (
                d_SumArrayOutput, d_OutputData, n);
            checkCUDAError("kernNaiveGPUScanThirdStep failed!");

            // cudaDeviceSynchronize();

            // Last step:
            convertFromInclusiveToExclusive <<<dimGridArray, dimBlockArray >>> (
                d_OutputData, d_OutputExclusiveData, n);
            checkCUDAError("convertFromInclusiveToExclusive failed!");
            timer().endGpuTimer();

            cudaMemcpy(odata, d_OutputExclusiveData, size, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            // cleanup
            cudaFree(d_InputData);
            cudaFree(d_OutputData);
            cudaFree(d_OutputExclusiveData);
            cudaFree(d_SumArray);
            cudaFree(d_SumArrayOutput);
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
