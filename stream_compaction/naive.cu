#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

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
 
        __global__ void kernNaiveGPUScan(int* inputArray, int* outputArray,
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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = n * sizeof(int);
            int* d_InputData;
            int* d_OutputData;

            cudaMalloc((void**)&d_InputData, size);
            checkCUDAError("cudaMalloc d_InputData failed!");

            cudaMalloc((void**)&d_OutputData, size);
            checkCUDAError("cudaMalloc d_OutputData failed!");

            cudaMemcpy(d_InputData, idata, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_OutputData, odata, size, cudaMemcpyHostToDevice);

            dim3 dimGrid((n + blockSize - 1) / blockSize, 1, 1);
            dim3 dimBlock(blockSize, 1, 1);

            timer().startGpuTimer();
            kernNaiveGPUScan <<<dimGrid, dimBlock>>> (d_InputData,
                d_OutputData, n);
            checkCUDAError("kernNaiveGPUScan failed!");
            timer().endGpuTimer();

            cudaMemcpy(odata, d_OutputData, size, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            // cleanup
            cudaFree(d_InputData);
            cudaFree(d_OutputData);
            checkCUDAError("cudaFree failed!");
        }
    }
}
