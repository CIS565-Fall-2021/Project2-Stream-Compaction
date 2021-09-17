#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

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
            
            // shift array for exclusive scan
            if (index + 1 >= n) {
                return;
            }
            // if first elem, should be 0
            if (index == 0) {
                odata[index] = 0;
            }

            int offset = pow(2, d-1);
            int nextIndex = index + 1;
            if (index >= offset){
                odata[nextIndex] = odata[nextIndex - offset] + idata[index];  
            }
            else{
                odata[nextIndex] = idata[index];   
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
            for (int d = 1; d < ilog2ceil(n); d++){
                kernScan << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_data2, dev_data1, d);
                
                // ping-pong
                int *tmp = dev_data1;
                dev_data1 = dev_data2;
                dev_data2 = tmp;
            }
            timer().endGpuTimer();
            // copy to odata to return
            cudaMemcpy(odata, dev_data2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }
    }
}
