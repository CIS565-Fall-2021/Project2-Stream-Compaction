#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

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
         * Performs prefix-sum (aka scan) on the buffer in place. Expects a padding to keep the length a power of 2.
         */
        void _scan(int n, int *dev_buf) {
            dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);

            // up-sweep phase
            for (int stride = 1; stride < n; stride <<= 1) {
              kernWorkEfficientUpSweepStep << <fullBlocksPerGrid, blockSize >> > (n, stride, dev_buf);
              checkCUDAError("kernWorkEfficientUpSweepStep failed!");
            }

            // down-sweep phase
            cudaMemset(&dev_buf[n - 1], 0, sizeof(int));
            for (int stride = n >> 1; stride > 0; stride >>= 1) {
              kernWorkEfficientDownSweepStep << <fullBlocksPerGrid, blockSize >> > (n, stride, dev_buf);
              checkCUDAError("kernWorkEfficientDownSweepStep failed!");
            }
        }
         
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          int nPow2 = 1 << ilog2ceil(n);

          int* dev_buf;
          cudaMalloc((void**)&dev_buf, sizeof(int) * nPow2);
          checkCUDAError("cudaMalloc dev_buf failed!");

          cudaMemcpy(dev_buf, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy to device failed!");

          timer().startGpuTimer();

          _scan(nPow2, dev_buf);

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy from device failed!");

          cudaFree(dev_buf);
          checkCUDAError("cudaFree dev_buf failed!");
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
            dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);

            int nPow2 = 1 << ilog2ceil(n);

            int* dev_input;
            cudaMalloc((void**)&dev_input, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_input failed!");

            cudaMemcpy(dev_input, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy input to device failed!");

            int* dev_bools;
            cudaMalloc((void**)&dev_bools, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_bools failed!");

            int* dev_indices;
            cudaMalloc((void**)&dev_indices, sizeof(int) * nPow2);
            checkCUDAError("cudaMalloc dev_indices failed!");

            timer().startGpuTimer();

            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_input);
            checkCUDAError("kernMapToBoolean failed!");

            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy from device to device failed!");

            _scan(nPow2, dev_indices);

            int count = 0;
            cudaMemcpy(&count, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from device failed!");
            count += idata[n - 1] != 0;

            int* dev_output;
            cudaMalloc((void**)&dev_output, sizeof(int) * count);
            checkCUDAError("cudaMalloc dev_output failed!");

            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_output, dev_input, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_output, sizeof(int) * count, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output to host failed!");

            cudaFree(dev_input);
            checkCUDAError("cudaFree dev_input failed!");

            cudaFree(dev_output);
            checkCUDAError("cudaFree dev_output failed!");

            cudaFree(dev_bools);
            checkCUDAError("cudaFree dev_bools failed!");

            cudaFree(dev_indices);
            checkCUDAError("cudaFree dev_indices failed!");

            return count;
        }
    }
}
