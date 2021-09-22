#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream> // testing 
#include <cassert> // for assert()

#define SECTION_SIZE 1024

namespace StreamCompaction {
    namespace Naive {
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

        __global__ void kernKoggeStoneScanAddUpSumArray(const int* S,
            int* Y, int inputSize)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize && blockIdx.x > 0)
            {
                Y[i] += S[blockIdx.x - 1];
            }
        }

        __global__ void kernKoggeStoneScan(int* X, int* Y, int* S, int inputSize)
        {
            __shared__ int XY[SECTION_SIZE];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize)
            {
                XY[threadIdx.x] = X[i];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            // performs iterative scan on XY
            // note that it is stride < blockDim.x, not stride <= blockDim.x: 
            // if you have 16 elements, stride could only be 1,2,4,8
            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
            {
                // make sure that input is in place
                __syncthreads();
                bool written = false;
                int temp = 0;
                if (threadIdx.x >= stride)
                {
                    temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
                    written = true;
                }
                // make sure previous output has been consumed
                __syncthreads();
                if (written)
                {
                    XY[threadIdx.x] = temp;
                }
            }
            Y[i] = XY[threadIdx.x];

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
        __global__ void kernKoggeStoneScan(int* X, int* Y, int inputSize)
        {
            __shared__ int XY[SECTION_SIZE];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize)
            {
                XY[threadIdx.x] = X[i];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            // performs iterative scan on XY
            // note that it is stride < blockDim.x, not stride <= blockDim.x: 
            // if you have 16 elements, stride could only be 1,2,4,8
            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
            {
                // make sure that input is in place
                __syncthreads();
                bool written = false;
                int temp = 0;
                if (threadIdx.x >= stride)
                {
                    temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
                    written = true;
                }
                // make sure previous output has been consumed
                __syncthreads();
                if (written)
                {
                    XY[threadIdx.x] = temp;
                }
            }
            Y[i] = XY[threadIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // n could be larger than SECTION_SIZE
            int idataSizeBytes = n * sizeof(int);

            int sumArraySizeBytes = (n / SECTION_SIZE) * sizeof(int);

            // MaxThreadsPerBlock: 1024
            assert(SECTION_SIZE <= 1024);
            assert(n <= 1048576); // 2^20

            dim3 dimGridKogge((n + SECTION_SIZE - 1) / SECTION_SIZE, 1, 1);
            dim3 dimBlockKogge(SECTION_SIZE, 1, 1);

            dim3 dimGridKoggeSumArray(1, 1, 1);
            dim3 dimBlockKoggeSumArray(SECTION_SIZE, 1, 1);

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
            kernKoggeStoneScan <<<dimGridKogge, dimBlockKogge >>> (d_X, d_Y, d_S, n);
            kernKoggeStoneScan <<<dimGridKoggeSumArray, dimBlockKoggeSumArray >>> (d_S, d_SOut, n);
            kernKoggeStoneScanAddUpSumArray <<<dimGridKogge, dimBlockKogge >>> (
                d_SOut, d_Y, n);
            convertFromInclusiveToExclusive << <dimGridKogge, dimBlockKogge >> > (
                d_Y, d_YExclusive, n);
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
    }
}

#if 0
void unitTestConversion()
{
    // for testing
    int numObject = 8;
    int size = numObject * sizeof(int);
    int* toyExclusiveArray = new int[numObject];
    int* toyInclusiveArray = new int[numObject] {3, 4, 11, 11, 15, 16, 22, 25};

    int* dev_toyExclusiveArray;
    int* dev_toyInclusiveArray;

    cudaMalloc((void**)&dev_toyExclusiveArray, size);
    checkCUDAError("cudaMalloc dev_toyExclusiveArray failed!");

    cudaMalloc((void**)&dev_toyInclusiveArray, size);
    checkCUDAError("cudaMalloc dev_toyInclusiveArray failed!");

    cudaMemcpy(dev_toyInclusiveArray, toyInclusiveArray, size,
        cudaMemcpyHostToDevice);

    dim3 dimGridArray((numObject + blockSize - 1) / blockSize, 1, 1);
    dim3 dimBlockArray(blockSize, 1, 1);
    convertFromInclusiveToExclusive << <dimGridArray, dimBlockArray >> > (
        dev_toyInclusiveArray, dev_toyExclusiveArray, numObject);

    cudaMemcpy(toyExclusiveArray, dev_toyExclusiveArray, size,
        cudaMemcpyDeviceToHost);
    checkCUDAError("memCpy back failed!");

    printf("\n");

    for (int i = 0; i < numObject; i++)
    {
        std::cout << toyExclusiveArray[i] << '\n';
    }

    printf("\n");

}
#endif