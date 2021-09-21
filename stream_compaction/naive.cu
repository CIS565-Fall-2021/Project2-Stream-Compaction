#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanNaive(int n, int depth, const int *dev_src, int *dev_dest) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n) return;

            int depthPow = 1 << (depth - 1);
            if (index >= depthPow) { // update curr
                dev_dest[index] = dev_src[index - depthPow] + dev_src[index];
            }
            else { // update from previous
                dev_dest[index] = dev_src[index];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_A;
            int* dev_B;
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_A!");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc failed on dev_B!");

            cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_B, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int maxDepth = ilog2ceil(n);
            for (int d = 1; d <= maxDepth; d++) {
                kernScanNaive << <fullBlocksPerGrid, blockSize >> > (n, d, dev_A, dev_B);
                std::swap(dev_A, dev_B);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_A, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost); // copy over with shift for exclusive scan
            odata[0] = 0; // set ident

            cudaFree(dev_A);
            checkCUDAErrorFn("cudaFree failed on dev_A!");
            cudaFree(dev_B);
            checkCUDAErrorFn("cudaFree failed on dev_B!");
        }
    }
}
