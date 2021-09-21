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

        __device__ inline int d_powi(int a, int b)
        {
            return (int)(powf(a, b) + 0.5f);
        }

        // parallel reduction
        __global__ void kernel_up_sweep(int* data, int d, int n)
        {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int leftInx = k + d_powi(2, d) - 1;
            int rightInx = k + d_powi(2, d + 1) - 1;

            if (rightInx > n - 1 || k % d_powi(2, d + 1) != 0) // note: this is an expensive check
                return;
            data[rightInx] += data[leftInx];
        }

        __global__ void kernel_down_sweep(int* data, int d, int n)
        {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int leftInx = k + d_powi(2, d) - 1;
            int rightInx = k + d_powi(2, d + 1) - 1;

            if (rightInx > n - 1 || k % d_powi(2, d+1) != 0) // note: this is an expensive check
                return;

            int leftChild = data[leftInx]; // save left child
            data[leftInx] = data[rightInx]; // set left child to this node's value
            data[rightInx] += leftChild; // set right child to old left value + 
                                         // this node's value
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        // TODO: padding doesn't seem to work...
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // allocate memory for the data buffer
            int* devData;
            int N = powi(2, ilog2ceil(n)); // get the minimum power of 2 >= nl

            cudaMalloc((void**)&devData, N * sizeof(int));

            // Copy idata to read and write buffer
            cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (N > n)
                cudaMemset(devData + n, 0, (N - n) * sizeof(int)); // padding 

            // define kernel dimension
            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

            // run up-sweep
            for (int d = 0; d <= ilog2ceil(N) - 1; d++)
            {
                kernel_up_sweep << <fullBlocksPerGrid, blockSize >> > (devData, d, n);
            }

            // run down-sweep
            cudaMemset(devData + N - 1, 0, sizeof(int));
            for (int d = ilog2ceil(N) - 1; d >= 0; d--)
            {
                kernel_down_sweep << <fullBlocksPerGrid, blockSize >> > (devData, d, n);
            }

            // Copy write buffer to odata
            cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(devData);
            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
