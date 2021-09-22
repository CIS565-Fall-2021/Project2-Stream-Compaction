#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"



namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_data1, * dev_data2;
            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data1 failed!");
            cudaMalloc((void**)&dev_data2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data2 failed!");
            cudaMemcpy(dev_data1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_data2, idata, sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            for (int i = 0; i < ilog2ceil(n); i++) {
                kernParallelScan << <fullBlocksPerGrid, blockSize >> > (n, i, dev_data1, dev_data2);
                checkCUDAError("kernParallelScan failed!");
                cudaDeviceSynchronize();
                int* temp = dev_data1;
                dev_data1 = dev_data2;
                dev_data2 = temp;
            }
            kernInclusiveToExclusive << <fullBlocksPerGrid, blockSize >> > (n, dev_data1, dev_data2);
            checkCUDAError("kernInclusiveToExclusive failed!");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data2, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }

        __global__ void kernParallelScan(int n, int level, int *src, int *dest) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < n) {
                if (index >= (1 << level)) {
                    dest[index] = src[index - (1 << level)] + src[index];
                }
                else {
                    dest[index] = src[index];
                }                
            }
        }

        __global__ void kernInclusiveToExclusive(int n, int *src, int *dest) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index == 0) {
                dest[index] = 0;
            } else if (index < n){
                dest[index] = src[index - 1];
            }
        }
    }
}

