#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int offset, int *buffer){
            // offset: current depth of the tree
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
	        if (index >= (N >> offset)) return;
            int k = index << (offset);
	        buffer[k + (1 << (offset)) - 1] += buffer[k + (1 << (offset-1)) - 1];
        }

        __global__ void kernDownSweep(int N, int offset, int *buffer){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= (N >> offset)) return;
            int k = index << offset;
            int tmp = buffer[k + (1 << offset) - 1];
            buffer[k + (1 << offset) - 1] += buffer[k + (1 << (offset - 1)) - 1];
            buffer[k + (1 << (offset - 1)) - 1] = tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *buffer, fullBlocksPerGrid;
            // padded to the power of 2s and get the max depth D of the balanced tree
            int D = ilog2ceil(n);
            int N = 1 << D;
            cudaMalloc((void**)&buffer, N * sizeof(int));
	        cudaMemcpy(buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
        
            for (int d= 0; d < D; d++) {
                fullBlocksPerGrid = ((N >> d) + blockSize - 1) / blockSize;
                kernUpSweep << <fullBlocksPerGrid, blockSize >> >(N, d+1, buffer);
            }
            timer().endGpuTimer();
            time += timer().getGpuElapsedTimeForPreviousOperation();
            cudaMemset(buffer + N - 1, 0, sizeof(int));

            timer().startGpuTimer();
            for (int d = D-1; d >= 0; d--) {
                fullBlocksPerGrid = ((N >> (d+1)) + blockSize - 1) / blockSize;
                kernDownSweep << <fullBlocksPerGrid, blockSize >> >(N, d+1, buffer);
            }    
            timer().endGpuTimer();
            time += timer().getGpuElapsedTimeForPreviousOperation();
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
            // Work-Efficient Compact
            time = 0;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            // init 
            int *bools, *indices, *in, *out;
            // memory allocation 
            cudaMalloc((void**)&bools, n * sizeof(int));
            cudaMalloc((void**)&indices, n * sizeof(int));
            cudaMalloc((void**)&in, n * sizeof(int));
            cudaMalloc((void**)&out, n * sizeof(int));
            // copy to device
            cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, bools, in);
            timer().endGpuTimer();
            time += timer().getGpuElapsedTimeForPreviousOperation();
            // copy to host
            cudaMemcpy(odata, bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            // work efficient scan
            scan(n, odata, odata);
            int lenCompacted = odata[n - 1];
            // copy to device
            cudaMemcpy(indices, odata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, out, in, bools, indices);
            timer().endGpuTimer();
            time += timer().getGpuElapsedTimeForPreviousOperation();
            printf("Work-Efficient scan: %f ms\n", time);
            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bools);
            cudaFree(indices);
            cudaFree(in);
            cudaFree(out);
            
            return lenCompacted;
        }
    }
}
