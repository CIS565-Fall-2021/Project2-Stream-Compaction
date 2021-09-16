#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128;
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
            
            int offset = pow(2, d-1);
            if (index >= offset){
                odata[index] = odata[index - offset] + idata[index];  
            }
            else{
                odata[index] = idata[index];   
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // allocate memory
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_data1 failed!");
            cudaMalloc((void**)&dev_data2, n * sizeof(int));
            
            timer().startGpuTimer();
           
            // map bools to ints
            kernMapToBoolean<<<fullBlocksPerGrid, threadsPerBlock>>>(n, idata, dev_data1);
            
            // for log iterations, perform scan
            for (int d = 1; d < ilog2ceil(n); d++){
                kernScan<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_data2, dev_data1, d);
                
                // ping-pong
                int *tmp = dev_data1;
                dev_data1 = dev_data2;
                dev_data2 = tmp;
            }
            timer().endGpuTimer();
            
            // cudaMemcpy dev_data2 to odata (device -> host)
        }
    }
}
