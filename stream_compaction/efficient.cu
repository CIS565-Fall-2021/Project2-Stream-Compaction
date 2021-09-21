#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 1024
//dim3 threadsPerBlock(blockSize);

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_array;

        //__global__ void kernReduction_1st_attempt(
        //    int array_length, int sum_ind_diff, int start_ind, int stride,
        //    int* array) {
        //    // compute one layer of scan in parallel.
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    if (index * stride + sum_ind_diff + start_ind >= array_length) {
        //        return;
        //    }
        //    array[index * stride + sum_ind_diff + start_ind] = array[index * stride + start_ind] + array[index * stride + sum_ind_diff + start_ind];
        //    __syncthreads();
        //}

        //__global__ void kernScanFromReduction_1st_attempt(
        //    int array_length, int sum_ind_diff, int start_ind, int stride,
        //    int* array) {
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    if (array_length-1 - index * stride - sum_ind_diff - start_ind < 0) {
        //        return;
        //    }
        //    int left_child = array[array_length - 1 - index * stride - sum_ind_diff];
        //    array[array_length - 1 - index * stride - sum_ind_diff] = array[array_length - 1 - index * stride];
        //    array[array_length - 1 - index * stride] = array[array_length - 1 - index * stride] + left_child;

        //}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        //void scan_1st_attempt(int n, int* odata, const int* idata) {
        //    // deal with non-2-power input
        //    int depth = ilog2ceil(n);
        //    int array_length = pow(2, depth);
        //    if (ilog2(n) != depth) {
        //        int* new_idata = new int[array_length];
        //        memset(new_idata, 0, array_length * sizeof(int));
        //        memcpy(new_idata, idata, n * sizeof(int));
        //        idata = new_idata;
        //    }
        //    cudaMalloc((void**)&dev_array, array_length * sizeof(int));
        //    cudaMemcpy(dev_array, idata, array_length * sizeof(int), cudaMemcpyHostToDevice);

        //    timer().startGpuTimer();
        //    dim3 fullBlocksPerGrid((array_length + blockSize - 1) / blockSize);
        //    for (int depth_ind = 0; depth_ind <= depth - 1; depth_ind++) {
        //        int sum_ind_diff = pow(2, depth_ind);
        //        int start_ind = sum_ind_diff - 1;
        //        int stride = pow(2, depth_ind + 1);
        //        kernReduction_1st_attempt << <fullBlocksPerGrid, blockSize >> > (array_length, sum_ind_diff, start_ind, stride, dev_array);
        //    }
        //    cudaDeviceSynchronize();

        //    cudaMemset(dev_array + array_length - 1, 0, sizeof(int));
        //    for (int depth_ind = depth - 1; depth_ind >=0 ; depth_ind--) {
        //        int sum_ind_diff = pow(2, depth_ind);
        //        int start_ind = sum_ind_diff - 1;
        //        int stride = pow(2, depth_ind + 1);
        //        kernScanFromReduction_1st_attempt << <fullBlocksPerGrid, blockSize >> > (array_length, sum_ind_diff, start_ind, stride, dev_array);
        //    }
        //    timer().endGpuTimer();
        //    cudaMemcpy(odata, dev_array, array_length * sizeof(int), cudaMemcpyDeviceToHost);

        //    //for (int ind = 0; ind < array_length; ind++) {
        //    //    printf("%d ", odata[ind]);
        //    //}
        //    //printf("\n");
        //    //printf("\n");
        //}

        __global__ void kernReduction(
            int array_length, int* array) {
            // compute scan in parallel.
            __shared__ int share_array[blockSize];
            int tx = threadIdx.x;
            if (tx >= array_length) {
                return;
            }
            share_array[tx] = array[tx];
            __syncthreads();
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                if (tx % (2 * stride) == (2 * stride) - 1) {
                    share_array[tx] += share_array[tx - stride];
                }
                __syncthreads();
            }
            array[tx] = share_array[tx];
        }

        //__global__ void kernReduction(
        //    int array_length, int* array) {
        //    // compute one layer of scan in parallel.
        //    int tx = threadIdx.x;
        //    if (tx >= array_length) {
        //        return;
        //    }
        //    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        //        if (tx % (2 * stride) == (2 * stride) - 1) {
        //            array[tx] += array[tx - stride];
        //        }
        //        __syncthreads();
        //    }
        //}

        //__global__ void kernScanFromReduction(
        //    int array_length, int depth, int* array) {
        //    int tx = threadIdx.x;
        //    if (tx >= array_length) {
        //        return;
        //    }
        //    for (int depth_ind = depth; depth_ind > 0; depth_ind--) {
        //        if (tx % (2 * stride) == (2 * stride) - 1) {
        //            array[tx] += array[tx - stride];
        //        }
        //        __syncthreads();
        //    }
        //}

        void scan(int n, int* odata, const int* idata) {
            int depth = ilog2ceil(n);
            int array_length = pow(2, depth);
            if (ilog2(n) != depth) {
                int* new_idata = new int[array_length];
                memset(new_idata, 0, array_length * sizeof(int));
                memcpy(new_idata, idata, n * sizeof(int));
                idata = new_idata;
            }
            cudaMalloc((void**)&dev_array, array_length * sizeof(int));
            cudaMemcpy(dev_array, idata, array_length * sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((array_length + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            kernReduction << <fullBlocksPerGrid, blockSize >> > (array_length, dev_array);
            cudaDeviceSynchronize();

            printf("%d ", depth);
            printf("\n");

            //cudaMemset(dev_array + array_length - 1, 0, sizeof(int));
            //kernScanFromReduction << <fullBlocksPerGrid, blockSize >> > (array_length, depth, dev_array);
            timer().endGpuTimer();
            //cudaMemcpy(odata, dev_array, array_length * sizeof(int), cudaMemcpyDeviceToHost);

            for (int ind = 0; ind < array_length; ind++) {
                printf("%d ", odata[ind]);
            }
            printf("\n");
            printf("\n");
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
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
