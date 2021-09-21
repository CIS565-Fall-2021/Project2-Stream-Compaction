#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int pow2, int depth, int *dev_data) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= pow2) return;
            
            int currSpacing = 1 << (depth + 1);
            if (index % currSpacing == 0) {
                dev_data[index + currSpacing - 1] += dev_data[index + (currSpacing >> 1) - 1];
            }
        }

        __global__ void kernDownSweep(int pow2, int depth, int *dev_data) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= pow2) return;

            int currSpacing = 1 << (depth + 1);
            int halfSpacing = 1 << depth;
            if (index % currSpacing == 0) {
                int temp = dev_data[index + halfSpacing - 1];
                dev_data[index + halfSpacing - 1] = dev_data[index + currSpacing - 1];
                dev_data[index + currSpacing - 1] += temp;
            }
        }



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int maxDepth = ilog2ceil(n);
            int pow2 = 1 << maxDepth;

            int *dev_data;
            cudaMalloc((void**)&dev_data, pow2 * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_data!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            bool startedLocal = false;
            if (!timer().getGpuTimerStarted()) {
                timer().startGpuTimer();
                startedLocal = true;
            }
            
            dim3 fullBlocksPerGrid((pow2 + blockSize - 1) / blockSize);
            for (int d = 0; d <= maxDepth - 1; d++) {
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (pow2, d, dev_data);
            }

            cudaMemset(&dev_data[pow2 - 1], 0, sizeof(int));
            for (int d = maxDepth - 1; d >= 0; d--) {
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (pow2, d, dev_data);
            }

            if (startedLocal) {
                timer().endGpuTimer();
            }

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            checkCUDAErrorFn("cudaFree failed on dev_data!");
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
            
            int* host_bools = new int[n];
            int out = 0;

            int* dev_odata;
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_odata!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_idata!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_bools!");
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_indices!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            int maxDepth = ilog2ceil(n);
            int pow2 = 1 << maxDepth;

            scan(n, dev_indices, dev_bools);

            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++) {
                if (host_bools[i] == 1) {
                    out++;
                }
            }

            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree failed on dev_odata!");
            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree failed on dev_idata!");
            cudaFree(dev_bools);
            checkCUDAErrorFn("cudaFree failed on dev_bools!");
            cudaFree(dev_indices);
            checkCUDAErrorFn("cudaFree failed on dev_indices!");

            return out;
        }
    }
}
