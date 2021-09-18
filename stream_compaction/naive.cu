#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // DONE: __global__
        __global__ void kernScanOnce(int n, int* odata, const int* idata, int stride) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int fromIdx = index - stride;
            int prevOdata = fromIdx < 0 ? 0 : idata[fromIdx];

            odata[index] = idata[index] + prevOdata;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int threadsPerBlock = 128;

            int* cuda_g_odata = nullptr;
            int* cuda_g_idata = nullptr;
            cudaMalloc(&cuda_g_odata, sizeof(int) * n);
            cudaMalloc(&cuda_g_idata, sizeof(int) * n);

            //cudaMemset(cuda_g_odata, 0, sizeof(odata) * n);
            cudaMemcpy(cuda_g_odata, idata, sizeof(int) * (n), cudaMemcpyHostToDevice);

            cudaDeviceSynchronize();
            //int logn = ilog2ceil(n);

            timer().startGpuTimer();
            // DONE
            //int stride = 1;
            //for (int i = 0; i < logn; ++i) {
            for(int stride = 1; stride <= n; stride <<= 1) {
                //int nextStride = stride << 1;
                int blockCount = (n + (threadsPerBlock - 1)) / threadsPerBlock;
                std::swap(cuda_g_odata, cuda_g_idata);
                //printf("stride:%d, blockCount:%d, threadsPerBlock:%d\n", stride, blockCount, threadsPerBlock);
                kernScanOnce<<<blockCount, threadsPerBlock>>>(n, cuda_g_odata, cuda_g_idata, stride);
                //stride = nextStride;
                //cudaDeviceSynchronize();
            }
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, cuda_g_odata, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_idata);
            cudaFree(cuda_g_odata);
        }
    }
}
