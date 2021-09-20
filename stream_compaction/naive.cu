#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream> // testing 

/*! Block size used for CUDA kernel launch. */
#define blockSize 128
#define sectionSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
 
        __global__ void kernNaiveGPUScanFirstStep(int* inputArray, int* outputArray, 
            int* SumArray, int inputSize)
        {
            // Each thread loads one value from the input array into shared 
            // memory array XY
            __shared__ int XY[sectionSize];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            // convert inclusive scan into exclusive scan by shifting 
            // all elements to the right by one position and fill the frist 
            // element and out-of-bound elements with 0. 
            if (i < inputSize && threadIdx.x != 0)
            {
                XY[threadIdx.x] = inputArray[i - 1];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            // perform naive scan
            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
            {
                // make sure that input is in place
                __syncthreads(); 
                int index = threadIdx.x;
                int previousIndex = index - stride;
                if (previousIndex < 0)
                {
                    previousIndex = 0;
                }
                int temp = XY[index] + XY[previousIndex];
                // make sure previous output has been consumed
                __syncthreads();
                XY[index] = temp;
            }

            // each thread writes its result into the output array
            outputArray[i] = XY[threadIdx.x];

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

        __global__ void kernNaiveGPUScanSecondStep(int* inputArray, int* outputArray,
            int inputSize)
        {
            // Each thread loads one value from the input array into shared 
            // memory array XY
            __shared__ int XY[sectionSize];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            // convert inclusive scan into exclusive scan by shifting 
            // all elements to the right by one position and fill the frist 
            // element and out-of-bound elements with 0. 
            if (i < inputSize && threadIdx.x != 0)
            {
                XY[threadIdx.x] = inputArray[i - 1];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            // perform naive scan
            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
            {
                // make sure that input is in place
                __syncthreads();
                int index = threadIdx.x;
                int previousIndex = index - stride;
                if (previousIndex < 0)
                {
                    previousIndex = 0;
                }
                int temp = XY[index] + XY[previousIndex];
                // make sure previous output has been consumed
                __syncthreads();
                XY[index] = temp;
            }

            // each thread writes its result into the output array
            outputArray[i] = XY[threadIdx.x];
        }

        __global__ void kernNaiveGPUScanThirdStep(int* inputArray, int* outputArray,
            int inputSize)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize && blockIdx.x > 0)
            {
                outputArray[i] += inputArray[blockIdx.x - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = n * sizeof(int);
            int* d_InputData;
            int* d_OutputData;
            int sumArrayNumEle = (n + blockSize - 1) / blockSize;
            int sumArraySize = sumArrayNumEle * sizeof(int);
            int* d_SumArray;

            // for testing
            int* sumArray = new int[sumArrayNumEle];

            cudaMalloc((void**)&d_InputData, size);
            checkCUDAError("cudaMalloc d_InputData failed!");

            cudaMalloc((void**)&d_OutputData, size);
            checkCUDAError("cudaMalloc d_OutputData failed!");

            cudaMalloc((void**)&d_SumArray, sumArraySize);
            checkCUDAError("cudaMalloc d_SumArray failed!");

            cudaMemcpy(d_InputData, idata, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_OutputData, odata, size, cudaMemcpyHostToDevice);

            dim3 dimGrid((n + blockSize - 1) / blockSize, 1, 1);
            dim3 dimBlock(blockSize, 1, 1);

            timer().startGpuTimer();
            // First step: compute the scan result for individual sections
            // then, store their block sum to sumArray
            kernNaiveGPUScanFirstStep <<<dimGrid, dimBlock>>> (d_InputData,
                d_OutputData, d_SumArray, n);
            checkCUDAError("kernNaiveGPUScanFirstStep failed!");
#if 0
            // cudaDeviceSynchronize();

            kernNaiveGPUScanFirstStep << <dimGrid, dimBlock >> > (d_InputData,
                d_OutputData, d_SumArray, n);
            checkCUDAError("kernNaiveGPUScanFirstStep failed!");

            // cudaDeviceSynchronize();

            kernNaiveGPUScanFirstStep << <dimGrid, dimBlock >> > (d_InputData,
                d_OutputData, d_SumArray, n);
            checkCUDAError("kernNaiveGPUScanFirstStep failed!");

            // cudaDeviceSynchronize();
#endif
            timer().endGpuTimer();

            cudaMemcpy(odata, d_OutputData, size, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            // testing: 
            cudaMemcpy(sumArray, d_SumArray, sumArraySize, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");
            for (int i = 0; i < sumArrayNumEle; i++)
            {
                std::cout << sumArray[i] << '\n';
            }
            printf("\n");
            for (int i = 0; i < n; i++)
            {
                std::cout << odata[i] << '\n';
            }
            

            // cleanup
            cudaFree(d_InputData);
            cudaFree(d_OutputData);
            checkCUDAError("cudaFree failed!");

            // testing clean up
            delete[] sumArray;
        }
    }
}
