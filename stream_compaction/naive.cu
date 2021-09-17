#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

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

            if (index >= (1 << d-1)) {
                odata[index] = idata[index - (1 << d - 1)] + idata[index];
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

            int paddedN = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_readable, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_writeable, paddedN * sizeof(int));

            cudaMemcpy(dev_readable, idata, paddedN * sizeof(int), cudaMemcpyHostToDevice);

            int numKernels = ilog2ceil(n);

            // --- begin iterative all-prefix-sum

            for (int d = 1; d <= log2(paddedN); d++) {

                // --- call scan ---

				kernNaiveScan <<<n, BLOCKSIZE>>> (paddedN, dev_writeable, dev_readable, d);
				checkCUDAErrorFn("naiveScan failed", "naive.cu", 63);
				cudaDeviceSynchronize();

                // --- ping pong buffers ---

                swp = dev_writeable;
                dev_writeable = dev_readable;
                dev_readable = swp;
            }

            timer().endGpuTimer();

            // this is an exclusive scan, so the first elem should be 0
            // and we shift everything (except the last elem) one index right
            odata[0] = 0;
            cudaMemcpy(odata+1, dev_readable, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_readable);
            cudaFree(dev_writeable);
        }
    }
}
