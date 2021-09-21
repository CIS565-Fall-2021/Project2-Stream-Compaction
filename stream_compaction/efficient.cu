#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
    namespace Efficient
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanEfficientUpSweep(int n, int layer, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            int shift = pow(2, layer);
            int validFlag = (((index + 1) & ((1 << (layer + 1)) - 1)) == 0 ? 1 : 0);
            // look at the right place, multiply by whether the bottom bits all 1
            int otherVal = validFlag * data[index - shift];
            __syncthreads();
            data[index] += otherVal;
        }

        __global__ void kernScanEfficientDownSweep(int n, int layer, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            int shift = pow(2, layer);
            int rChildFlag = ((index & ((1 << (layer + 1)) - 1)) == 0 ? 1 : 0);
            int lChildFlag = !rChildFlag &&
                             ((index & ((1 << (layer)) - 1)) == 0 ? 1 : 0);
            // int lVal = rChildFlag * data[index - shift];
            // int rVal = lChildFlag * data[index + shift];
            int nextVal =
                // case is right child, case not child covered
                (lChildFlag == 0) * (data[index] + (rChildFlag * data[index - shift])) +
                lChildFlag * data[index + shift];
            __syncthreads();
            data[index] = nextVal;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            //next power of 2
            int power = ilog2ceil(n);
            int size = pow(2, power);
            int offset = size - n;
            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
            int *buf;
            cudaMalloc((void **)&buf, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc buf failed!");
            cudaMemset(buf, 0, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMemset buf failed!");
            cudaMemcpy(buf + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy buf failed!");
            timer().startGpuTimer();
            // TODO
            // for (int layer = 0; layer < power; layer++)
            for (int layer = 0; layer < 1; layer++)
            {
                // invoke kernel
                kernScanEfficientUpSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, buf);
                cudaDeviceSynchronize();
                checkCUDAErrorWithLine("cudaDeviceSynchronize buf failed!");
            }
            // cudaMemset(buf + size - 1, 0, 1 * sizeof(int));
            // checkCUDAErrorWithLine("cudaMemset buf failed!");
            // for (int layer = power - 1; layer >= 0; layer--)
            // {
            //     // invoke kernel
            //     kernScanEfficientDownSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, buf);
            //     cudaDeviceSynchronize();
            //     checkCUDAErrorWithLine("cudaDeviceSynchronize buf failed!");
            // }
            timer().endGpuTimer();
            cudaMemcpy(odata, buf + offset, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy buf failed!");
            // odata[0] = 0;
            cudaFree(buf);
            checkCUDAErrorWithLine("cudaFree buf failed!");
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
        int compact(int n, int *odata, const int *idata)
        {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
