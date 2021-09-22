#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int d, int n, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d + 1)) == 0) {
                idata[index + (1 << (d + 1)) - 1] += idata[index + (1 << d) - 1];
            }
        }


        __global__ void kernDownSweep(int d, int n, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d + 1)) == 0) {
                int t = idata[index + (1 << d) - 1];
                idata[index + (1 << d) - 1] = idata[index + (1 << (d + 1)) - 1];
                idata[index + (1 << (d + 1)) - 1] += t;
            }
        }


        __global__ void kernSetZero(int n, int* idata) {
            idata[n - 1] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool timing) {

            //pad to power of 2
            int paddedSize = 1 << ilog2ceil(n);

            int* deviceIn;
            cudaMalloc((void**)&deviceIn, paddedSize * sizeof(int));

            dim3 fullBlocksPerGrid((paddedSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

            //From index n to paddedSize are 0s.
            cudaMemcpy(deviceIn, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            if (timing) {
                timer().startGpuTimer();
            }

            //Up sweep
            for (int d = 0; d <= ilog2ceil(paddedSize) - 1; d++) {
                kernUpSweep << < fullBlocksPerGrid, BLOCK_SIZE >> > (d, paddedSize, deviceIn);
                checkCUDAError("kernUpSweep failed");
            }

            //Down sweep 
            kernSetZero << < 1, 1 >> > (paddedSize, deviceIn);
            for (int d = ilog2ceil(paddedSize) - 1; d >= 0; d--) {
                kernDownSweep << < fullBlocksPerGrid, BLOCK_SIZE >> > (d, paddedSize, deviceIn);
                checkCUDAError("kernDownSweep failed");
            }

            if (timing) {
                timer().endGpuTimer();
            }

            cudaMemcpy(odata, deviceIn, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(deviceIn);
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
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* count = new int[2];

            int* deviceIn;
            cudaMalloc((void**)&deviceIn, n * sizeof(int));
            int* deviceBool;
            cudaMalloc((void**)&deviceBool, n * sizeof(int));
            int* deviceBoolPSum;
            cudaMalloc((void**)&deviceBoolPSum, n * sizeof(int));
            int* deviceOut;
            cudaMalloc((void**)&deviceOut, n * sizeof(int));

            cudaMemcpy(deviceIn, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            kernMapToBoolean << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, deviceBool, deviceIn);
            checkCUDAError("kernMapToBoolean failed!");

            scan(n, deviceBoolPSum, deviceBool, false);

            kernScatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, deviceOut, deviceIn, deviceBool, deviceBoolPSum);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            cudaMemcpy(count, deviceBool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(count + 1, deviceBoolPSum + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            //size equals to last of boolean array and last of boolean prefix sum array
            int compactedSize = count[0] + count[1];

            cudaMemcpy(odata, deviceOut, sizeof(int) * compactedSize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            cudaFree(deviceIn);
            cudaFree(deviceBool);
            cudaFree(deviceBoolPSum);
            cudaFree(deviceOut);

            return compactedSize;
        }
    }
}
