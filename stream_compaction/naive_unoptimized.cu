#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 1024
dim3 threadsPerBlock(blockSize);

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        int* dev_array;
        
        __global__ void kernScanLayer(
            int array_length, int stride, int* array) {
            // compute one layer of scan in parallel.
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= array_length - stride) {
                return;
            }
            array[index + stride] = array[index] + array[index + stride];
            __syncthreads();
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // deal with non-2-power input
            int depth = ilog2ceil(n);
            int array_length = pow(2, depth);
            if (ilog2(n) != depth) {
                int* new_idata = new int[array_length];
                memset(new_idata, 0, array_length * sizeof(int));
                memcpy(new_idata, idata, n * sizeof(int));
                idata = new_idata;
            }

            dim3 fullBlocksPerGrid((array_length + blockSize - 1) / blockSize);
            cudaMalloc((void**)&dev_array, array_length * sizeof(int));
            cudaMemcpy(dev_array + 1, idata, (array_length - 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_array, 0, 1);

            timer().startGpuTimer();
            for (int depth_ind = 1; depth_ind <= depth; depth_ind++) {
                int stride = pow(2, depth_ind - 1);
                kernScanLayer << <fullBlocksPerGrid, blockSize >> > (array_length, stride, dev_array);
                
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_array, array_length * sizeof(int), cudaMemcpyDeviceToHost);

            //int* array_0 = new int[array_length];
            //int* array_1 = new int[array_length];
            //cudaMemcpy(array_0, dev_array_dep1, array_length * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(array_1, dev_array_dep2, array_length * sizeof(int), cudaMemcpyDeviceToHost);
            //printf("\n");
            //printf("\n");
            //for (int ind = 0; ind < array_length; ind++) {
            //    printf("%d ", array_0[ind]);
            //}
            //printf("\n");
            //printf("\n");
            //for (int ind = 0; ind < array_length; ind++) {
            //    printf("%d ", array_1[ind]);
            //}
            //printf("\n");
            //printf("\n");
            
            
        }
    }
}
