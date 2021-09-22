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

        int* dev_odata;
        int* dev_idata;

        __global__ void kernPrefixSum(int n, int offset, int *out, const int *in) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n) {
            return;
          }

          out[index] = in[index];
          if (index >= offset)
            out[index] += in[index - offset];
        }

        __global__ void kernShiftRight(int n, int* odata, const int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n)
            return;

          // make exclusive
          odata[index] = (index > 0) ? idata[index - 1] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

          // TODO: see if there's better way to force power of 2
          int N = imakepower2(n);

          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          cudaMalloc((void **)&dev_odata, sizeof(int) * N);
          checkCUDAErrorFn("dev_odata malloc failed.");
          cudaMalloc((void **)&dev_idata,  sizeof(int) * N);
          checkCUDAErrorFn("dev_idata malloc failed.");

          cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          checkCUDAErrorFn("memCpy idata to dev_idata failed.");

          timer().startGpuTimer();

          for (int d = 1; d < N; d *= 2) {
            // Copy dev_odata to dev_idata (to be used as input for the next iteration)
            kernPrefixSum<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_odata, dev_idata);

            int* tmp = dev_idata;
            dev_idata = dev_odata;
            dev_odata = tmp;
          }

          timer().endGpuTimer();
          checkCUDAErrorFn("kernPrefixSum failed.");
          
          cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);  
          checkCUDAErrorFn("memCpy dev_odata1 to odata failed.");

          cudaFree(dev_odata);
          cudaFree(dev_idata);
          checkCUDAErrorFn("cudaFree failed.");
        }
    }
}
