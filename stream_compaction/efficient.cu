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

        __global__ void kernel_bitmap(int* input, int* output, int n)
        {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (k > n - 1)
                return;

            output[k] = input[k] > 0;
        }

        __global__ void kernel_scatter(int* input, int* bitmap, int* scanResult, int* dest, int n)
        {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (k > n - 1)
                return;

            if (bitmap[k] == 1)
            {
                dest[scanResult[k]] = input[k];
            }
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
            timer().startGpuTimer();

            // allocate memory for the data buffer
            int* devBuffer;
            int N = powi(2, ilog2ceil(n)); // get the minimum power of 2 >= n

            cudaMalloc((void**)&devBuffer, N * sizeof(int));

            // Copy idata to read and write buffer
            cudaMemcpy(devBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (N > n)
                cudaMemset(devBuffer + n, 0, (N - n) * sizeof(int)); // padding 

            // run efficient scan algorithm
            efficientScan(devBuffer, N);

            // Copy write buffer to odata
            cudaMemcpy(odata, devBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(devBuffer);

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

            // step 1: compute temporary bitmap array
            kernel_bitmap << <fullBlocksPerGrid, blockSize >> > (devInput, devBitmap, N);

            // step 2: run exclusive scan on temporary bitmap array
            cudaMemcpy(devScan, devBitmap, N * sizeof(int), cudaMemcpyDeviceToDevice);
            efficientScan(devScan, N);

            // step 3: scatter
            kernel_scatter << <fullBlocksPerGrid, blockSize >> > (devInput, devBitmap, devScan, devOutput, N);
            int numOfElements = 0;
            cudaMemcpy(&numOfElements, devScan + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            //if (odata[n-1] == 0)
            //    numOfElements--; // exclusive scan, may need to disregard last element

            // copy device's output buffer to odata
            cudaMemcpy(odata, devOutput, numOfElements * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(devInput);
            cudaFree(devBitmap);
            cudaFree(devScan);
            cudaFree(devOutput);

            timer().endGpuTimer();
            return numOfElements;
        }
    }
}
