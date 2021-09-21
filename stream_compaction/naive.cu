#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream> // testing 
#include <cassert> // for assert()



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

        __device__ void computeScanToOutputArray(const int* inputArray, int* outputArray,
            int* XY, int inputSize)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < inputSize)
            {
                XY[threadIdx.x] = inputArray[i];
            }
            else {
                XY[threadIdx.x] = 0;
            }
            // perform naive scan
            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
            {
                // make sure that input is in place
                __syncthreads();
                int previousValue = 0;
                int previousIndex = threadIdx.x - stride;
                if (previousIndex >= 0)
                {
                    previousValue = XY[previousIndex];
                }
                int temp = XY[threadIdx.x] + previousValue;
                // make sure previous output has been consumed
                __syncthreads();
                XY[threadIdx.x] = temp;
            }

            // each thread writes its result into the output array
            outputArray[i] = XY[threadIdx.x];
        }
        
        __global__ void kernNaiveGPUScanFirstStep(const int* inputArray, 
            int* outputArray, int* SumArray, int inputSize)
        {
            // Each thread loads one value from the input array into shared 
            // memory array XY
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

        __global__ void kernNaiveGPUScanSecondStep(const int* inputArray, 
            int* outputArray, int inputSize)
        {
            // Each thread loads one value from the input array into shared 
            // memory array XY
            __shared__ int XY[MAX_SUM_ARRAY_SIZE];
            computeScanToOutputArray(inputArray, outputArray, XY, inputSize);
        }

        __global__ void kernNaiveGPUScanThirdStep(const int* inputArray, 
            int* outputArray, int inputSize)
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
            int sumArrayNumEle = (n + blockSize - 1) / blockSize;
            assert(sumArrayNumEle <= 1024 && "Sum Array has more than 1024 elements!");
            int sumArraySize = sumArrayNumEle * sizeof(int);

            int* d_InputData;
            int* d_OutputData;
            int* d_OutputExclusiveData;
            int* d_SumArray;
            int* d_SumArrayOutput;
            int* d_SumArrayAx;

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

            cudaMalloc((void**)&d_SumArrayAx, sumArraySize);
            checkCUDAError("cudaMalloc d_SumArrayOutput failed!");

            cudaMemcpy(d_InputData, idata, size, cudaMemcpyHostToDevice);

            dim3 dimGridArray((n + blockSize - 1) / blockSize, 1, 1);
            dim3 dimBlockArray(blockSize, 1, 1);

            
            dim3 dimGridSumArray(1, 1, 1);
            dim3 dimBlockSumArray(sumArrayNumEle, 1, 1);

            // for testing
            int* sumArray = new int[sumArrayNumEle];
            int* sumArrayOutput = new int[sumArrayNumEle];

            timer().startGpuTimer();
            // First step: compute the scan result for individual sections
            // then, store their block sum to sumArray
            kernNaiveGPUScanFirstStep << <dimGridArray, dimBlockArray >> > (d_InputData,
                d_OutputData, d_SumArray, n);
            checkCUDAError("kernNaiveGPUScanFirstStep failed!");
#if 0
            cudaDeviceSynchronize();
            cudaMemcpy(odata, d_OutputData, size, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            cudaMemcpy(sumArray, d_SumArray, sumArraySize, cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            std::cout << '\n';
            for (int i = 0; i < n; i++)
            {
                std::cout << odata[i] << ' ';
                if ((i + 1) % 8 == 0) {
                    std::cout << std::endl;
                }
            }

            std::cout << '\n';
            for (int i = 0; i < sumArrayNumEle; i++)
            {
                std::cout << sumArray[i] << ' ';
            }

            std::cout << '\n';
#endif
            // Second step: scan block sums
            kernNaiveGPUScanSecondStep << <dimGridSumArray, dimBlockSumArray >> > (
                d_SumArray, d_SumArrayOutput, sumArrayNumEle);
            checkCUDAError("kernNaiveGPUScanSecondStep failed!");
#if 0

            cudaMemcpy(sumArrayOutput, d_SumArrayOutput, sumArraySize,
                cudaMemcpyDeviceToHost);
            checkCUDAError("memCpy back failed!");

            printf("\n");

            for (int i = 0; i < sumArrayNumEle; i++)
            {
                std::cout << sumArrayOutput[i] << ' ';
            }

            printf("\n");

#endif
            // Third step: add scanned block sum i to all values of scanned block
            // i + 1
            kernNaiveGPUScanThirdStep << <dimGridArray, dimBlockArray >> > (
                d_SumArrayOutput, d_OutputData, n);
            checkCUDAError("kernNaiveGPUScanThirdStep failed!");

           // cudaDeviceSynchronize();

            // Last step:

            convertFromInclusiveToExclusive << <dimGridArray, dimBlockArray >> > (
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