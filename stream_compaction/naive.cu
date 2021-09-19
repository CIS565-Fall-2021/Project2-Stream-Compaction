#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernResetBuffer(int nPadded, int* dataPadded1, int* dataPadded2) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return;
            }

            // reset memory 
            dataPadded1[index] = 0; 
            dataPadded2[index] = 0;
        }

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

            int offset = 1 << depth;
            // copy old values that won't be computed
            if (index < offset) {
                dataPadded2[index] = dataPadded1[index]; 
                return; 
            }

            // compute new values
            dataPadded2[index] = dataPadded1[index - offset] + dataPadded1[index];
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
            
            // instantiate blocks 
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid(nPadded / threadsPerBlock.x + 1);

            // reset idata buffer to 0, reset odata buffer to quiet_NaN
            kernResetBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, dev_dataPadded1, dev_dataPadded2);
            
            // copy idata to device memory 
            cudaMemcpy(dev_dataPadded1, idata, n * sizeof(int), cudaMemcpyHostToDevice); 
            cudaMemcpy(dev_dataPadded2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // cudaDeviceSynchronize(); 

            // begin scan process
            timer().startGpuTimer();
            for (int i = 0; i < depth; i++) {
                // perform partial scan on depth i
                kernScanNaive<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_dataPadded1, dev_dataPadded2);
                // swap to avoid race conditions
                std::swap(dev_dataPadded1, dev_dataPadded2);
            }

            // make scan exclusive
            kernExclusive<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, dev_dataPadded1, dev_dataPadded2);
            // cudaDeviceSynchronize(); 

            // copy scan back to host
            cudaMemcpy(odata, dev_dataPadded2, n * sizeof(int), cudaMemcpyDeviceToHost); 
            timer().endGpuTimer();
            
            // free local buffers
            cudaFree(dev_dataPadded1);
            cudaFree(dev_dataPadded2); 
        }
    }
}
