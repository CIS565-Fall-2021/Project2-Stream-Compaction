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
            int otherVal = ((index & ((1 << (layer + 1)) - 1)) == 0 ? 1 : 0) *
                           data[index - shift];
            __syncthreads();
            data[index] += otherVal;
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
            cudaMemcpy(buf + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO
            for (int layer = 0; layer < power; layer++)
            {
                // invoke kernel
                kernScanEfficientUpSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, buf);
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            cudaMemcpy(odata + 1, buf + offset, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
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
