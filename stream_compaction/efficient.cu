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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int round = 1 << ilog2ceil(n);
            int* dev_data;
            cudaMalloc((void**)&dev_data, round * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            for (int i = 0; i < ilog2(round); i++) {
                int numThreads = (round >> (i + 1));
                dim3 fullBlocksPerGrid(numThreads / efficientBlockSize);
                if (fullBlocksPerGrid.x == 0) {
                    kernUpSweep << <1, numThreads >> > (round, i, dev_data);
                }
                else {
                    kernUpSweep << <fullBlocksPerGrid, efficientBlockSize >> > (round, i, dev_data);
                }                
                checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }
            kernSetRootZero << <1, 1 >> > (round, dev_data);
            for (int i = ilog2(round) - 1; i >= 0; i--) {
                int numThreads = (round >> (i + 1));
                dim3 fullBlocksPerGrid(numThreads / efficientBlockSize);
                if (fullBlocksPerGrid.x == 0) {
                    kernDownSweep << <1, numThreads >> > (round, i, dev_data);
                }
                else {
                    kernDownSweep << <fullBlocksPerGrid, efficientBlockSize >> > (round, i, dev_data);
                }                
                checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        __global__ void kernUpSweep(int n, int level, int* arr) {
            int index = (blockIdx.x * blockDim.x + threadIdx.x) << (level + 1);
            if (index < n) {
                arr[index + (1 << (level + 1)) - 1] += arr[index + (1 << level) - 1];
            }
        }
        

        __global__ void kernSetRootZero(int n, int* arr) {
            arr[n - 1] = 0;
        }

        __global__ void kernDownSweep(int n, int level, int* arr) {
            int index = (blockIdx.x * blockDim.x + threadIdx.x) << (level + 1);
            if (index < n) {
                int left = arr[index + (1 << level) - 1];
                arr[index + (1 << level) - 1] = arr[index + (1 << (level + 1)) - 1];
                arr[index + (1 << (level + 1)) - 1] += left;
            }
        }

        __global__ void kernScanShared(int n, int logn, int* arr, int* sums) {
            __shared__ int sArr[2 * efficientBlockSize];
            int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            if (index < n) {
                sArr[2 * threadIdx.x] = arr[index];
                sArr[2 * threadIdx.x + 1] = arr[index + 1];
                for (int i = 0; i < logn; i++) {
                    __syncthreads();
                    if (threadIdx.x < (blockDim.x >> i)) {                        
                        sArr[(threadIdx.x << (i + 1)) + (1 << (i + 1)) - 1] += sArr[(threadIdx.x << (i + 1)) + (1 << i) - 1];
                    }                    
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    sums[blockIdx.x] = sArr[2 * blockDim.x - 1];
                    sArr[2 * blockDim.x - 1] = 0;
                }
                for (int i = logn - 1; i >= 0; i--) {
                    __syncthreads();
                    if (threadIdx.x < (blockDim.x >> i)) {
                        int left = sArr[(threadIdx.x << (i + 1)) + (1 << i) - 1];
                        sArr[(threadIdx.x << (i + 1)) + (1 << i) - 1] = sArr[(threadIdx.x << (i + 1)) + (1 << (i + 1)) - 1];
                        sArr[(threadIdx.x << (i + 1)) + (1 << (i + 1)) - 1] += left;
                    }
                }
                __syncthreads();
                arr[index] = sArr[2 * threadIdx.x];
                arr[index + 1] = sArr[2 * threadIdx.x + 1];
                __syncthreads();
            }
        }


        void optimizedScan(int n, int* odata, const int* idata) {
            int round = 1 << ilog2ceil(n);
            int* dev_data;
            cudaMalloc((void**)&dev_data, round * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            optimizedScanRecursive(round, dev_data);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }


        void optimizedScanRecursive(int n, int* dev_data) {
            int round = 1 << ilog2ceil(n);
            dim3 fullBlocksPerGrid(round / (2 * efficientBlockSize));
            int* dev_sum;
            if (fullBlocksPerGrid.x == 0) {
                int logn = ilog2(round);
                cudaMalloc((void**)&dev_sum, 1 * sizeof(int));
                kernScanShared << <1, round / 2>> > (round, logn, dev_data, dev_sum);
            }
            else {
                cudaMalloc((void**)&dev_sum, fullBlocksPerGrid.x * sizeof(int));
                int logn = ilog2(efficientBlockSize) + 1;
                kernScanShared << <fullBlocksPerGrid, efficientBlockSize >> > (round, logn, dev_data, dev_sum);
            }
            cudaDeviceSynchronize();
            if (fullBlocksPerGrid.x > 1) {
                optimizedScanRecursive(fullBlocksPerGrid.x, dev_sum);
                cudaDeviceSynchronize();
                fullBlocksPerGrid.x = (round / efficientBlockSize);
                kernBlockIncrement << <fullBlocksPerGrid, efficientBlockSize >> > (round, dev_data, dev_sum);
                cudaDeviceSynchronize();                
            }   
            cudaFree(dev_sum);

        }

        __global__ void kernBlockIncrement(int n, int* data, int* increment) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ int offset;
            if (threadIdx.x == 0) {
                offset = increment[index / (2 * efficientBlockSize)];
            }
            __syncthreads();
            data[index] += offset;

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
            int round = 1 << ilog2ceil(n);
            int* dev_scan, *dev_bool, *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_scan, round * sizeof(int));
            checkCUDAError("cudaMalloc dev_scan failed!");
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + efficientBlockSize - 1) / efficientBlockSize);
            Common::kernMapToBoolean << <fullBlocksPerGrid, efficientBlockSize >> > (n, dev_bool, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");
            cudaMemcpy(dev_scan, dev_bool, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            for (int i = 0; i < ilog2(round); i++) {
                int numThreads = (round >> (i + 1));
                dim3 fullBlocksPerGrid(numThreads / efficientBlockSize);
                if (fullBlocksPerGrid.x == 0) {
                    kernUpSweep << <1, numThreads >> > (round, i, dev_scan);
                }
                else {
                    kernUpSweep << <fullBlocksPerGrid, efficientBlockSize >> > (round, i, dev_scan);
                }
                checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }
            kernSetRootZero << <1, 1 >> > (round, dev_scan);
            for (int i = ilog2(round) - 1; i >= 0; i--) {
                int numThreads = (round >> (i + 1));
                dim3 fullBlocksPerGrid(numThreads / efficientBlockSize);
                if (fullBlocksPerGrid.x == 0) {
                    kernDownSweep << <1, numThreads >> > (round, i, dev_scan);
                }
                else {
                    kernDownSweep << <fullBlocksPerGrid, efficientBlockSize >> > (round, i, dev_scan);
                }
                checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }
            Common::kernScatter << <fullBlocksPerGrid, efficientBlockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_scan);
            checkCUDAError("kernScatter failed!");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int count, lastBool;
            cudaMemcpy(&count, &dev_scan[n-1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, &dev_bool[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_scan);
            cudaFree(dev_bool);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            return count + lastBool;
        }

        int optimizedCompact(int n, int* odata, const int* idata) {
            int round = 1 << ilog2ceil(n);
            int* dev_scan, * dev_bool, * dev_idata, * dev_odata;
            cudaMalloc((void**)&dev_scan, round * sizeof(int));
            checkCUDAError("cudaMalloc dev_scan failed!");
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + efficientBlockSize - 1) / efficientBlockSize);
            Common::kernMapToBoolean << <fullBlocksPerGrid, efficientBlockSize >> > (n, dev_bool, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");
            cudaMemcpy(dev_scan, dev_bool, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            optimizedScanRecursive(n, dev_scan);
            Common::kernScatter << <fullBlocksPerGrid, efficientBlockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_scan);
            checkCUDAError("kernScatter failed!");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int count, lastBool;
            cudaMemcpy(&count, &dev_scan[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, &dev_bool[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_scan);
            cudaFree(dev_bool);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            return count + lastBool;
        }
    }
}
