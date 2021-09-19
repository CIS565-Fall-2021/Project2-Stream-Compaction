#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

#define blockSize 128
dim3 threadsPerBlock(blockSize);

int *dev_data1;
int *dev_data2;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernScan(int n, int *odata, int *idata, int d){
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            int offset = pow(2, d - 1);

            // if d == 1, shift array over to accomodate first 0
            int outIndex = d == 1 ? index + 1 : index;

            // if outIndex >= n, return (skips last elem on first run)
            if (outIndex >= n) {
                return;
            }
            // if first elem, should be 0
            if (index == 0 && d == 1) {
                odata[index] = 0;
            }
 
            if (index >= offset){
                odata[outIndex] = idata[index - offset] + idata[index];
            }
            else{
                odata[outIndex] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // allocate memory
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            cudaMalloc((void**)&dev_data2, n * sizeof(int));
            cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            
            // for log iterations, perform scan
            for (int d = 1; d < ilog2ceil(n) + 1; d++){
                kernScan << <fullBlocksPerGrid, n >> > (n, dev_data2, dev_data1, d);

                // FOR TESTING ONLY
                /*cudaMemcpy(odata, dev_data2, n * sizeof(int), cudaMemcpyDeviceToHost);
                std::cout << "Round: " << d << ", [" << odata[0] << ", " << odata[1] << ", " << odata[2] << ", " << odata[3] << ", "
                    << odata[4] << ", " << odata[5] << ", " << odata[6] << ", " << odata[7] << ", "
                    << odata[8] << ", " << odata[9] << "]" << std::endl;*/
                
                // ping-pong
                int *tmp = dev_data1;
                dev_data1 = dev_data2;
                dev_data2 = tmp;
            }
            timer().endGpuTimer();
            // copy to odata to return
            cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }
    }
}
