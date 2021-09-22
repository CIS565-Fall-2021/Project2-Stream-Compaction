#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

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

        void efficientScan(int* cudaBuffer, int n)
        {
            // define kernel dimension
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // run up-sweep
            for (int d = 0; d <= ilog2ceil(n) - 1; d++)
            {
                kernel_up_sweep << <fullBlocksPerGrid, blockSize >> > (cudaBuffer, d, n);
            }

            // run down-sweep
            cudaMemset(cudaBuffer + n - 1, 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--)
            {
                kernel_down_sweep << <fullBlocksPerGrid, blockSize >> > (cudaBuffer, d, n);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // allocate memory for the data buffer
            int* devBuffer;
            int N = powi(2, ilog2ceil(n)); // get the minimum power of 2 >= n

            cudaMalloc((void**)&devBuffer, N * sizeof(int));

            // Copy idata to read and write buffer
            cudaMemcpy(devBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (N > n)
                cudaMemset(devBuffer + n, 0, (N - n) * sizeof(int)); // padding 

            // START TIMER
            timer().startGpuTimer();

            // run efficient scan algorithm
            efficientScan(devBuffer, N);

            timer().endGpuTimer();
            // TIMER END

            // Copy write buffer to odata
            cudaMemcpy(odata, devBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(devBuffer);
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
            int *devInput, *devBitmap, * devScan, * devOutput;
            int N = powi(2, ilog2ceil(n)); // get the minimum power of 2 >= n

            // allocate memory to device buffers
            cudaMalloc((void**)&devInput, N * sizeof(int));
            cudaMalloc((void**)&devBitmap, N * sizeof(int));
            cudaMalloc((void**)&devScan, N * sizeof(int));
            cudaMalloc((void**)&devOutput, N * sizeof(int));

            // copy idata to device buffer
            cudaMemcpy(devInput, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (N > n)
                cudaMemset(devInput + n, 0, (N - n) * sizeof(int)); // padding 

            // define kernel dimension
            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

            // START TIMER
            timer().startGpuTimer();

            // step 1: compute temporary bitmap array
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (N, devBitmap, devInput);

            // step 2: run exclusive scan on temporary bitmap array
            cudaMemcpy(devScan, devBitmap, N * sizeof(int), cudaMemcpyDeviceToDevice);
            efficientScan(devScan, N);

            // step 3: scatter
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (N, devOutput, devInput, devBitmap, devScan);

            timer().endGpuTimer();
            // TIMER END

            // find number of elements
            int numOfElements = 0;
            cudaMemcpy(&numOfElements, devScan + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            // copy device's output buffer to odata
            cudaMemcpy(odata, devOutput, numOfElements * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(devInput);
            cudaFree(devBitmap);
            cudaFree(devScan);
            cudaFree(devOutput);
            return numOfElements;
        }
    }
}
