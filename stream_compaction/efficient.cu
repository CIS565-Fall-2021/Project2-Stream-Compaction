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

        __global__ void kernScanEfficientUpSweep(int n, int layer, int shift, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            if (index % (1 << (layer + 1)) == 0)
            {
                data[index + (1 << (layer + 1)) - 1] += data[index + (1 << layer) - 1];
            }
        }

        __global__ void kernScanEfficientDownSweep(int n, int layer, int max, int shift, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            if (index % (1 << (layer + 1)) == 0)
            {
                int t = data[index + (1 << layer) - 1];
                data[index + (1 << layer) - 1] = data[index + (1 << (layer + 1)) - 1];
                data[index + (1 << (layer + 1)) - 1] += t;
            }
        }

        __global__ void kernSetLastToZero(int n, int *data)
        {
            data[n - 1] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            //next power of 2
            int power = ilog2ceil(n);
            int size = 1 << power;
            int offset = size - n;
            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
            int *buf;
            cudaMalloc((void **)&buf, size * sizeof(int));
            cudaMemset(buf, 0, size * sizeof(int));
            cudaMemcpy(buf + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO
            for (int layer = 0; layer < power; layer++)
            {
                // invoke kernel
                int shift = 1 << layer;
                kernScanEfficientUpSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, shift, buf);
                cudaDeviceSynchronize();
            }
            kernSetLastToZero<<<1, 1>>>(size, buf);
            cudaDeviceSynchronize();
            for (int layer = power - 1; layer >= 0; layer--)
            {
                // invoke kernel
                int shift = 1 << layer;
                kernScanEfficientDownSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, power - 1, shift, buf);
                cudaDeviceSynchronize();
                // checkCUDAErrorWithLine("cudaDeviceSynchronize buf failed!");
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, buf + offset, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buf);
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
            //next power of 2
            int power = ilog2ceil(n);
            int size = 1 << power;
            int offset = size - n;
            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

            int *buf; // power of 2 0 padded copy of idata
            cudaMalloc((void **)&buf, size * sizeof(int));
            cudaMemset(buf, 0, size * sizeof(int));
            cudaMemcpy(buf + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int *bools, *indices, *tmpOut; //
            cudaMalloc((void **)&bools, size * sizeof(int));
            cudaMalloc((void **)&indices, size * sizeof(int));
            cudaMalloc((void **)&tmpOut, size * sizeof(int));
            timer().startGpuTimer();
            // TODO
            // Map
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(size, bools, buf);
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(size, indices, buf);
            cudaDeviceSynchronize();
            // Scan
            for (int layer = 0; layer < power; layer++)
            {
                // invoke kernel
                int shift = 1 << layer;
                kernScanEfficientUpSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, shift, indices);
                cudaDeviceSynchronize();
            }
            kernSetLastToZero<<<1, 1>>>(size, indices);
            cudaDeviceSynchronize();
            for (int layer = power - 1; layer >= 0; layer--)
            {
                // invoke kernel
                int shift = 1 << layer;
                kernScanEfficientDownSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, power - 1, shift, indices);
                cudaDeviceSynchronize();
            }
            // Scatter
            Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(size, tmpOut, buf, bools, indices);
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, tmpOut, n * sizeof(int), cudaMemcpyDeviceToHost);
            int retSize;
            cudaMemcpy(&retSize, indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int tmpLast;
            cudaMemcpy(&tmpLast, buf + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            retSize += (tmpLast != 0);

            cudaFree(buf);
            cudaFree(bools);
            cudaFree(indices);
            cudaFree(tmpOut);
            return retSize;
        }
    }
}
