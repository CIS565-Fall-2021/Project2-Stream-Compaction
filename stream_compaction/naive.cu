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
        // TODO: (DONE)
        __global__ void kernNaiveScan(int N, int* odata, int* idata, int offset) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx >= N) {
                return;
            }
            if (idx >= offset) {
                odata[idx] = idata[idx - offset] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO (DONE)
            const int blockSize = 128;
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            int max = ilog2ceil(n);

            //buffers
            int* buf1;
            int* buf2;

            //malloc
            cudaMalloc((void**)&buf1, n * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc buf1 failed!");
            cudaMalloc((void**)&buf2, n * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc buf2 failed!");

            //fill array
            cudaMemcpy(buf1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            //call kernel
            timer().startGpuTimer();
            for (int i = 1; i <= max; i++) {
                kernNaiveScan<<<blocksPerGrid, blockSize>>>(n, buf2, buf1, (int)powf(2, i - 1));
                std::swap(buf1, buf2);
            }
            timer().endGpuTimer();

            //copy data to odata
            odata[0] = 0;
            cudaMemcpy(odata + 1, buf1, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(buf1);
            cudaFree(buf2);
        }
    }
}
