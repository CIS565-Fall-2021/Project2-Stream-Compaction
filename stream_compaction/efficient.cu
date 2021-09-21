#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweepIter(int nPadded, int depth, int* dataPadded) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return; 
            }

            int offset = 1 << (depth + 1); 

            if (index % offset == 0) {
                dataPadded[index + offset - 1] += dataPadded[index + (offset >> 1) - 1];
            }
        }

        __global__ void kernDownSweepIter(int nPadded, int depth, int* dataPadded) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return;
            }

            int offset = 1 << (depth + 1); 
            if (index % offset == 0) {
                int temp = dataPadded[index + (offset >> 1) - 1];
                dataPadded[index + (offset >> 1) - 1] = dataPadded[index + offset - 1];
                dataPadded[index + offset - 1] += temp;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            if (n < 1) { return; }

            // allocate a buffer padded to a power of 2. 
            int depth = ilog2ceil(n);
            int nPadded = 1 << depth;

            int* dev_dataPadded;
            cudaMalloc((void**)&dev_dataPadded, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataPadded failed!");

            // set blocks and threads 
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid(std::ceil((double)nPadded / blockSize));

            // copy idata to device memory 
            cudaMemset(dev_dataPadded, 0, nPadded * sizeof(int));
            checkCUDAError("cudaMemset dev_dataPadded failed!");
            cudaMemcpy(dev_dataPadded, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_dataPadded failed!");

            timer().startGpuTimer();
            // TODO
            // perform upsweep on idata
            for (int i = 0; i < depth; i++) {
                kernUpSweepIter<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_dataPadded); 
                checkCUDAError("kernUpSweepIter failed!");
            }

            // perform downsweep on idata
            cudaMemset(dev_dataPadded + nPadded - 1, 0, sizeof(int)); 
            checkCUDAError("cudaMemset dev_dataPadded + nPadded - 1 failed!");
            for (int i = depth - 1; i >= 0; i--) {
                kernDownSweepIter<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_dataPadded); 
                checkCUDAError("kernDownSweepIter failed!");
            }

            cudaDeviceSynchronize();
            checkCUDAError("cudaDeviceSynchronize failed!");
            timer().endGpuTimer();

            // copy scan back to host
            cudaMemcpy(odata, dev_dataPadded, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_dataPadded failed!");

            cudaFree(dev_dataPadded); 
            checkCUDAError("cudaFree dev_dataPadded failed!");
        }

        __global__ void kernScatter(int nPadded, const int* idata, int* odata, const int* dataPadded) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= nPadded) {
                return;
            }

            if (idata[index]) {
                odata[dataPadded[index]] = idata[index]; 
            }
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
            // TODO

            if (n < 1) { return -1; }

            // allocate a buffer padded to a power of 2. 
            int depth = ilog2ceil(n);
            int nPadded = 1 << depth;

            // calling kernels means we cannot directly index into idata. Need to have a device copy 
            int* dev_dataPadded;
            cudaMalloc((void**)&dev_dataPadded, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataPadded failed!");
            cudaMemset(dev_dataPadded, 0, n * sizeof(int)); 
            checkCUDAError("cudaMemset dev_dataPadded failed!"); 
            cudaMemcpy(dev_dataPadded, idata, n * sizeof(int), cudaMemcpyHostToDevice); 
            checkCUDAError("cudaMemcpy dev_dataPadded failed!"); 

            // mapping of true and false for idata
            int* dev_bools;
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            // array that will be scanned into 
            int* dev_index; 
            cudaMalloc((void**)&dev_index, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_index failed!");
            cudaMemset(dev_index, 0, nPadded * sizeof(int));
            checkCUDAError("cudaMemset dev_index failed!");

            int* dev_odata; 
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            // set blocks and threads 
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid(std::ceil((double) nPadded / blockSize));

            timer().startGpuTimer();

            // SCAN
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_bools, dev_index, dev_dataPadded);
            checkCUDAError("kernMapToBoolean failed!");



            // perform upsweep on idata
            for (int i = 0; i < depth; i++) {
                kernUpSweepIter << <fullBlocksPerGrid, threadsPerBlock >> > (nPadded, i, dev_index);
                checkCUDAError("kernUpSweepIter failed!");
            }

            // perform downsweep on idata
            cudaMemset(dev_index + nPadded - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_dataPadded + nPadded - 1 failed!");
            for (int i = depth - 1; i >= 0; i--) {
                kernDownSweepIter<<<fullBlocksPerGrid, threadsPerBlock>>>(nPadded, i, dev_index);
                checkCUDAError("kernDownSweepIter failed!");
            }

            // SCATTER
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_dataPadded, dev_bools, dev_index);
            checkCUDAError("kernScatter failed!");
            cudaDeviceSynchronize();
            checkCUDAError("cudaDeviceSynchronize failed!");

            timer().endGpuTimer();
            
            // return compact to odata
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_bools failed!");

            // return final index and bool to host to calculate number of elements
            int idx, val; 
            cudaMemcpy((void*)&idx, dev_index + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy idx failed!");
            cudaMemcpy((void*)&val, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy val failed!");

            // free 
            cudaFree(dev_dataPadded);
            checkCUDAError("cudaFree dev_dataPadded failed!");
            cudaFree(dev_bools);
            checkCUDAError("cudaFree dev_bools failed!");
            cudaFree(dev_index);
            checkCUDAError("cudaFree dev_index failed!");
            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed!");

            return idx + val;
        }
    }
}
