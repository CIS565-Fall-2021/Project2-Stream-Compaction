#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256
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
        int* dev_array_static;

        int* dev_idata;
        int* dev_odata;
        int* dev_bools;
        int* dev_indices;
        

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
            int array_length, int start_ind, int* array) {
            // compute scan in parallel.
            __shared__ int share_array[blockSize];
            int tx = threadIdx.x;
            if (tx >= array_length) {
                return;
            }
            share_array[tx] = array[start_ind + tx];
            __syncthreads();
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                if (tx % (2 * stride) == (2 * stride) - 1) {
                    share_array[tx] += share_array[tx - stride];
                }
                __syncthreads();
            }
            array[start_ind + tx] = share_array[tx];
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

        __global__ void kernScanFromReduction(
            int array_length, int depth, int start_ind, int* array) {
            __shared__ int share_array[blockSize];
            int tx = threadIdx.x;
            if (tx >= array_length) {
                return;
            }
            //if (blockSize <= array_length) {
            //    if (tx == blockSize - 1) {
            //        share_array[tx] = 0;
            //    }
            //    else {
            //        share_array[tx] = array[start_ind + tx];
            //    }
            //}
            //else {
            //    if (tx == array_length - 1) {
            //        share_array[tx] = 0;
            //    }
            //    else {
            //        share_array[tx] = array[start_ind + tx];
            //    }
            //}
            if (tx == blockSize - 1) {
                share_array[tx] = 0;
            }
            else {
                share_array[tx] = array[start_ind + tx];
            }
            __syncthreads();
            for (int depth_ind = depth - 1; depth_ind >= 0; depth_ind--) {
                int stride = pow(2, depth_ind);
                if (tx % (2 * stride) == (2 * stride) - 1) {
                    int left_child = share_array[tx - stride];
                    share_array[tx - stride] = share_array[tx];
                    share_array[tx] += left_child;
                }
                __syncthreads();
            }
            __syncthreads();
            // convert result to inclusive
            if (tx != blockSize - 1) {
                array[start_ind + tx] = share_array[tx + 1];
            }
        }

        __global__ void kernAdd(
            int array_length, int value_ind, int* array_static, int start_ind, int* array) {
            int tx = threadIdx.x;
            __shared__ int value;
            value = array_static[value_ind];
            if (tx >= array_length) {
                return;
            }
            array[tx + start_ind] += value;
        }

        //__global__ void kernScanFromReduction(
        //    int array_length, int depth, int* array) {
        //    int tx = threadIdx.x;
        //    if (tx >= array_length) {
        //        return;
        //    }
        //    for (int depth_ind = depth-1; depth_ind >= 0; depth_ind--) {
        //        int stride = pow(2, depth_ind);
        //        if (tx % (2 * stride) == (2 * stride) - 1) {
        //            int left_child = array[tx - stride];
        //            array[tx - stride] = array[tx];
        //            array[tx] += left_child;
        //        }
        //        __syncthreads();
        //    }
        //}

        void scan(int n, int* odata, const int* idata, bool timer_on) {
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

            int num_block;
            if (array_length < blockSize) {
                num_block = 1;
            }
            else {
                num_block = array_length / blockSize;
            }

            if (timer_on) {
                timer().startGpuTimer();
            }
            for (int block_ind = 0; block_ind < num_block; block_ind++) {
                int start_ind = block_ind * blockSize;
                kernReduction << <fullBlocksPerGrid, blockSize >> > (array_length, start_ind, dev_array);
            }
            cudaDeviceSynchronize();

            for (int block_ind = 0; block_ind < num_block; block_ind++) {
                int start_ind = block_ind * blockSize;
                kernScanFromReduction << <fullBlocksPerGrid, blockSize >> > (array_length, depth, start_ind, dev_array);
            }
            cudaDeviceSynchronize();

            cudaMalloc((void**)&dev_array_static, array_length * sizeof(int));
            cudaMemcpy(dev_array_static, dev_array, array_length * sizeof(int), cudaMemcpyHostToDevice);
            for (int block_ind2 = 1; block_ind2 < num_block; block_ind2++) {
                for (int block_ind = block_ind2; block_ind < num_block; block_ind++) {
                    int start_ind = block_ind * blockSize;
                    int value_ind = block_ind2 * blockSize - 1;
                    kernAdd << <fullBlocksPerGrid, blockSize >> > (array_length, value_ind, dev_array_static, start_ind, dev_array);
                }
                cudaDeviceSynchronize();
            }
            if (timer_on) {
                timer().endGpuTimer();
            }
            cudaMemcpy(odata + 1, dev_array, (array_length - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            //printf("\n");
            //printf("\n");

            //for (int ind = 0; ind < array_length / 2; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");

            //for (int ind = array_length / 2; ind < array_length; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");

            //for (int ind = 0; ind < array_length/4; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");

            //for (int ind = array_length / 4; ind < array_length/2; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");

            //for (int ind = array_length / 2; ind < array_length / 4 * 3; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");

            //for (int ind = array_length / 4 * 3; ind < array_length; ind++) {
            //    printf("%d ", odata[ind]);
            //}
            //printf("\n");
            //printf("\n");
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
            int depth = ilog2ceil(n);
            int array_length = pow(2, depth);
            if (ilog2(n) != depth) {
                int* new_idata = new int[array_length];
                memset(new_idata, 0, array_length * sizeof(int));
                memcpy(new_idata, idata, n * sizeof(int));
                idata = new_idata;
            }
            cudaMalloc((void**)&dev_bools, array_length * sizeof(int));
            cudaMalloc((void**)&dev_indices, array_length * sizeof(int));
            cudaMalloc((void**)&dev_idata, array_length * sizeof(int));
            cudaMalloc((void**)&dev_odata, array_length * sizeof(int));
            cudaMemcpy(dev_idata, idata, array_length * sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((array_length + blockSize - 1) / blockSize);
            int count = 0;
            int* host_bools = (int*) malloc(array_length * sizeof(int));
            int num_block;
            if (array_length < blockSize) {
                num_block = 1;
            }
            else {
                num_block = array_length / blockSize;
            }

            timer().startGpuTimer();
            for (int block_ind = 0; block_ind < num_block; block_ind++) {
                int start_ind = block_ind * blockSize;
                Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (array_length, dev_bools + start_ind, dev_idata + start_ind);
            }
            cudaDeviceSynchronize();

            cudaMemcpy(host_bools, dev_bools, array_length * sizeof(int), cudaMemcpyDeviceToHost);
            Efficient::scan(array_length, odata, host_bools, false);
            //for (int ind = 0; ind < array_length; ind++) {
            //    printf("%d ", host_bools[ind]);
            //}
            //printf("\n");
            //printf("\n");
            cudaMemcpy(dev_indices, odata, array_length * sizeof(int), cudaMemcpyHostToDevice);

            for (int block_ind = 0; block_ind < num_block; block_ind++) {
                int start_ind = block_ind * blockSize;
                Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (array_length, dev_odata, dev_idata + start_ind, dev_bools + start_ind, dev_indices + start_ind);
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            cudaMemcpy(&count, dev_indices + array_length - 1, 1 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            return count;
        }
    }
}
