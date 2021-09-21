#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <vector>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScanStep(int n, int offset, int* odata, const int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= 0 && index < n) {
            if (index >= offset) {
              odata[index] = idata[index - offset] + idata[index];
            }
            else {
              odata[index] = idata[index];
            }
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);

            int* dev_buf0;
            cudaMalloc((void**)&dev_buf0, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buf0 failed!");

            int* dev_buf1;
            cudaMalloc((void**)&dev_buf1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buf0 failed!");
            
            cudaMemcpy(dev_buf0, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            for (int offset = 1; offset < n; offset <<= 1) {
              kernNaiveScanStep << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_buf1, dev_buf0);
              checkCUDAError("kernNaiveScanStep failed!");

              std::swap(dev_buf0, dev_buf1);
            }

            cudaMemcpy(&odata[1], dev_buf0, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from device failed!");

            std::vector<int> odataView(n);
            memcpy(odataView.data(), odata, sizeof(int) * n);

            std::vector<int> idataView(n);
            memcpy(idataView.data(), idata, sizeof(int) * n);

            cudaFree(dev_buf0);
            checkCUDAError("cudaFree dev_buf0 failed!");

            cudaFree(dev_buf1);
            checkCUDAError("cudaFree dev_buf1 failed!");

            timer().endGpuTimer();
        }
    }
}
