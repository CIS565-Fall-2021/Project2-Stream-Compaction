#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <vector>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernWorkEfficientUpSweepStep(int n, int stride, int* data) {
          int index = 2 * stride * (threadIdx.x + (blockIdx.x * blockDim.x)) - 1;
          if (index >= stride && index < n) {
            data[index] += data[index - stride];
          }
        }
        
        __global__ void kernWorkEfficientDownSweepStep(int n, int stride, int* data) {
          int index = 2 * stride * (threadIdx.x + (blockIdx.x * blockDim.x)) - 1;
          if (index >= stride && index < n) {
            int oldValue = data[index];
            data[index] += data[index - stride];
            data[index - stride] = oldValue;
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int nPow2 = 1 << ilog2ceil(n);
            dim3 fullBlocksPerGrid = ((nPow2 + blockSize - 1) / blockSize);

            int* dev_buf;
            cudaMalloc((void**)&dev_buf, sizeof(int) * nPow2);
            checkCUDAError("cudaMalloc dev_buf failed!");

            cudaMemcpy(dev_buf, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            // up-sweep phase
            for (int stride = 1; stride < nPow2; stride <<= 1) {
              kernWorkEfficientUpSweepStep << <fullBlocksPerGrid, blockSize >> > (nPow2, stride, dev_buf);
              checkCUDAError("kernWorkEfficientUpSweepStep failed!");
            }

            // down-sweep phase
            cudaMemset(&dev_buf[nPow2 - 1], 0, sizeof(int));
            for (int stride = nPow2 >> 1; stride > 0; stride >>= 1) {
              kernWorkEfficientDownSweepStep << <fullBlocksPerGrid, blockSize >> > (nPow2, stride, dev_buf);
              checkCUDAError("kernWorkEfficientDownSweepStep failed!");
            }

            cudaMemcpy(odata, dev_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from device failed!");

            cudaFree(dev_buf);
            checkCUDAError("cudaFree dev_buf failed!");

            std::vector<int> odataView(n);
            memcpy(odataView.data(), odata, sizeof(int) * n);

            std::vector<int> idataView(n);
            memcpy(idataView.data(), idata, sizeof(int) * n);

            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
