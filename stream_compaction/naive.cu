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

        __global__ void kernExclusive(int nPadded, const int* dataPadded1, int* dataPadded2) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return;
            }

            // if first element, pad with identity. Otherwise copy left element
            dataPadded2[index] = (index) ? dataPadded1[index - 1] : 0;
        }

        __global__ void kernScanNaive(int nPadded, int depth, const int* dataPadded1, int* dataPadded2) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return;
            }

            // copy old values that won't be computed
            if (index < depth) {
                dataPadded2[index] = dataPadded1[index]; 
                return; 
            }

            // compute new values
            dataPadded2[index] = dataPadded1[index - depth] + dataPadded1[index];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            if (n < 1) { return;  }

            // allocate two buffers padded to a power of 2. 
            int depth = ilog2ceil(n); 
            int nPadded = 1 << depth;

            int* dev_dataPadded1; int* dev_dataPadded2;
            cudaMalloc((void**)&dev_dataPadded1, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataExtended1 failed!");
            cudaMalloc((void**)&dev_dataPadded2, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataExtended2 failed!");
            
            // set blocks and threads 
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid(std::ceil((double) nPadded / blockSize));
            
            // copy idata to device memory 
            cudaMemset(dev_dataPadded1, 0, nPadded * sizeof(int));
            checkCUDAError("cudaMemset dev_dataPadded1 failed!");
            cudaMemcpy(dev_dataPadded1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_dataPadded1 failed!");

            // begin scan process
            timer().startGpuTimer();
            for (int i = 1; i < nPadded; i <<= 1) {
                // perform partial scan on depth i
                kernScanNaive<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_dataPadded1, dev_dataPadded2);
                // swap to avoid race conditions
                std::swap(dev_dataPadded1, dev_dataPadded2);
            }

            // make scan exclusive
            kernExclusive<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, dev_dataPadded1, dev_dataPadded2);
            cudaDeviceSynchronize(); 
            checkCUDAError("cudaDeviceSynchronize failed!");
            timer().endGpuTimer();

            // copy scan back to host
            cudaMemcpy(odata, dev_dataPadded2, n * sizeof(int), cudaMemcpyDeviceToHost); 
            checkCUDAError("cudaMemcpy dev_dataPadded2 failed!");

            // free local buffers
            cudaFree(dev_dataPadded1);
            checkCUDAError("cudaFree dev_dataPadded1 failed!");
            cudaFree(dev_dataPadded2);
            checkCUDAError("cudaFree dev_dataPadded2 failed!");
        }
    }
}
