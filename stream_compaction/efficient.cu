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

        __global__ void kernScanUpSweepPhase(int threads, int *dev_temp, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= threads) {
                return;
            }
            index = (index + 1) * offset * 2 - 1;
            dev_temp[index] += dev_temp[index - offset];
        }

        __global__ void kernScanDownSweepPhase(int threads, int *dev_temp, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= threads) {
                return;
            }
            index = (index + 1) * offset * 2 - 1;
            int t = dev_temp[index - offset];
            dev_temp[index - offset] = dev_temp[index];
            dev_temp[index] += t;
        }

        void scanHelper(int steps, int size, int *dev_temp) {

            int threads = size / 2;
            int offset = 1;
            for (int i = 0; i < steps; ++i) {
                dim3 blocks((threads + blockSize - 1) / blockSize);
                kernScanUpSweepPhase<<<blocks, blockSize>>>(threads, dev_temp, offset);
                threads /= 2;
                offset *= 2;
            }

            cudaMemset(dev_temp + size - 1, 0, sizeof(int));
            threads = 1;
            offset = size / 2;
            for (int i = 0; i < steps; ++i) {
                dim3 blocks((threads + blockSize - 1) / blockSize);
                kernScanDownSweepPhase<<<blocks, blockSize>>>(threads, dev_temp, offset);
                threads *= 2;
                offset /= 2;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int steps = ilog2ceil(n);
            int size = 1 << steps;

            int *dev_temp;
            cudaMalloc((void**) &dev_temp, size * sizeof(int));
            cudaMemcpy(dev_temp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_temp + n, 0, (size - n) * sizeof(int));

            timer().startGpuTimer();
            scanHelper(steps, size, dev_temp);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_temp, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_temp);
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

            dim3 blocks((n + blockSize - 1) / blockSize);
            int steps = ilog2ceil(n);
            int size = 1 << steps;

            int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
            cudaMalloc((void**) &dev_idata, n * sizeof(int));
            cudaMalloc((void**) &dev_odata, n * sizeof(int));
            cudaMalloc((void**) &dev_bools, n * sizeof(int));
            cudaMalloc((void**) &dev_indices, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_indices + n, 0, (size - n) * sizeof(int));

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanHelper(steps, size, dev_indices);
            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

            int lastBool, lastIdx;
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIdx, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return lastBool + lastIdx;
        }

        void radixSort(int n, int *odata, int *idata) {

            int *dev_idata, *dev_odata, *dev_bools, *dev_scan;
            cudaMalloc((void**) &dev_idata, n * sizeof(int));
            cudaMalloc((void**) &dev_odata, n * sizeof(int));
            cudaMalloc((void**) &dev_bools, n * sizeof(int));
            cudaMalloc((void**) &dev_scan, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int maxNum = 0;
            for (int i = 0; i < n; ++i) {
                maxNum = std::max(maxNum, idata[n]);
            }

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_scan);
        }
    }
}
