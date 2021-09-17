#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <cmath>

#ifndef BLOCKSIZE
#define BLOCKSIZE 128
#endif // !BLOCKSIZE


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int *odata, const int *idata, int d) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) {
                return;
            }

            if (index >= pow(2, d-1)) {
                odata[index] = idata[index - (int)pow(2, d - 1)] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
            __syncthreads();

            return;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            //int* odata2;
            int* dev_readable; 
            int* dev_writeable; 
            int* swp; // for ping-ponging odata and odata2

            //int p2Pad = ilog2ceil(n);

            cudaMalloc((void**)&dev_readable, n * sizeof(int));
            cudaMalloc((void**)&dev_writeable, n * sizeof(int));

            cudaMemcpy(dev_readable, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int numKernels = ilog2ceil(n);

            // --- begin iterative all-prefix-sum

            for (int d = 1; d <= log2(n); d++) {

                // --- call scan ---

				kernNaiveScan <<<n, BLOCKSIZE>>> (n, dev_writeable, dev_readable, d);
				checkCUDAErrorFn("naiveScan failed", "naive.cu", 63);
				cudaDeviceSynchronize();

                // --- ping pong buffers ---

                swp = dev_writeable;
                dev_writeable = dev_readable;
                dev_readable = swp;
            }

            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata+1, dev_readable, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_readable);
            cudaFree(dev_writeable);
        }
    }
}
