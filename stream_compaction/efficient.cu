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
            int validFlag = !((index + 1) & ((1 << (layer + 1)) - 1));
            // look at the right place, multiply by whether the bottom bits all 1
            int otherVal = validFlag * data[index - shift];
            __syncthreads();
            data[index] += otherVal;
        }

        __global__ void kernScanEfficientDownSweep(int n, int layer, int max, int shift, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            // int rChildFlag = (((index + 1) & ((1 << (layer + 1)) - 1)) == 0 ? 1 : 0);
            // int lChildFlag = !rChildFlag &&
            //                  (((index + 1) & ((1 << (layer)) - 1)) == 0 ? 1 : 0);
            int rChildFlag = !((index + 1) & ((1 << (layer + 1)) - 1));
            int lChildFlag = !rChildFlag &&
                             !((index + 1) & ((1 << (layer)) - 1));
            int nextVal = !lChildFlag * (data[index] + (rChildFlag * data[index - shift])) +
                          lChildFlag * data[index + shift];
            // nextVal *= !((index == n - 1) && (layer == max));
            __syncthreads();
            data[index] = nextVal;
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
            for (int layer = 0; layer < power; layer++)
            {
                // invoke kernel
                int shift = pow(2, layer);
                kernScanEfficientUpSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, shift, buf);
                cudaDeviceSynchronize();
                // checkCUDAErrorWithLine("cudaDeviceSynchronize buf failed!");
            }
            kernSetLastToZero<<<1, 1>>>(size, buf);
            cudaDeviceSynchronize();
            for (int layer = power - 1; layer >= 0; layer--)
            {
                // invoke kernel
                int shift = pow(2, layer);
                kernScanEfficientDownSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, power - 1, shift, buf);
                cudaDeviceSynchronize();
                // checkCUDAErrorWithLine("cudaDeviceSynchronize buf failed!");
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, buf + offset, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy buf failed!");
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
            //next power of 2
            int power = ilog2ceil(n);
            int size = pow(2, power);
            int offset = size - n;
            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

            int *buf; // power of 2 0 padded copy of idata
            cudaMalloc((void **)&buf, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc buf failed!");
            cudaMemset(buf, 0, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMemset buf failed!");
            cudaMemcpy(buf + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy buf failed!");

            int *bools, *indices, *tmpOut; //
            cudaMalloc((void **)&bools, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc bools failed!");
            cudaMalloc((void **)&indices, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc indices failed!");
            cudaMalloc((void **)&tmpOut, size * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc tmpOut failed!");
            timer().startGpuTimer();
            // TODO
            // Map
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(size, bools, buf);
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(size, indices, buf);
            cudaDeviceSynchronize();
            checkCUDAErrorWithLine("cudaDeviceSynchronize failed!");
            // Scan
            for (int layer = 0; layer < power; layer++)
            {
                // invoke kernel
                int shift = pow(2, layer);
                kernScanEfficientUpSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, shift, indices);
                cudaDeviceSynchronize();
                checkCUDAErrorWithLine("cudaDeviceSynchronize failed!");
            }
            kernSetLastToZero<<<1, 1>>>(size, indices);
            cudaDeviceSynchronize();
            checkCUDAErrorWithLine("cudaDeviceSynchronize failed!");
            for (int layer = power - 1; layer >= 0; layer--)
            {
                // invoke kernel
                int shift = pow(2, layer);
                kernScanEfficientDownSweep<<<fullBlocksPerGrid, blockSize>>>(size, layer, power - 1, shift, indices);
                cudaDeviceSynchronize();
                checkCUDAErrorWithLine("cudaDeviceSynchronize failed!");
            }
            // Scatter
            Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(size, tmpOut, buf, bools, indices);
            cudaDeviceSynchronize();
            checkCUDAErrorWithLine("cudaDeviceSynchronize failed!");
            timer().endGpuTimer();

            cudaMemcpy(odata, tmpOut + offset, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy failed!");
            cudaFree(buf);
            checkCUDAErrorWithLine("cudaFree buf failed!");
            cudaFree(bools);
            checkCUDAErrorWithLine("cudaFree bools failed!");
            cudaFree(indices);
            checkCUDAErrorWithLine("cudaFree indices failed!");
            cudaFree(tmpOut);
            checkCUDAErrorWithLine("cudaFree tmpOut failed!");
            return -1;
        }
    }
}
