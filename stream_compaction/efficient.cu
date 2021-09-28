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

        /*for d = 0 to log2n - 1
            for all k = 0 to n 每 1 by 2^(d + 1) in parallel
                x[k + 2^(d + 1) 每 1] += x[k + 2^d 每 1];*/
        __global__ void kernUpSweep(int* data, int d, int maxSize) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index > maxSize) {
                return;
            }
            int powD = powf(2.0, d);
            int powDplusOne = powf(2.0, d + 1);

            int selected = index * powDplusOne;

            if (selected >= maxSize) {
                return;
            }

            data[selected + powDplusOne - 1] += data[selected + powD - 1];

    

        }


        //x[n - 1] = 0
        //    for d = log2n 每 1 to 0
        //        for all k = 0 to n 每 1 by 2d + 1 in parallel
        //            t = x[k + 2d 每 1];               // Save left child
        //            x[k + 2d 每 1] = x[k + 2d + 1 每 1];  // Set left child to this node＊s value
        //            x[k + 2d + 1 每 1] += t;             // Set right child to old left value +
        //                                 // this node＊s value

        __global__ void kernDownSweep(int* data, int d, int maxSize) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index > maxSize) {
                return;
            }

            int powD = powf(2.0, d);
            int powDplusOne = powf(2.0, d + 1);

            int selected = index * powDplusOne;

            if (selected >= maxSize) {
                return;
            }

            int temp = data[selected + powD - 1];
            data[selected + powD - 1] = data[selected + powDplusOne - 1];
            data[selected + powDplusOne - 1] = temp + data[selected + powDplusOne - 1];

      

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {


            int totalD = ilog2ceil(n);
            int maxSize = pow(2, totalD);
            int blockSize = 128;
            dim3 fullBlocksPerGrid((maxSize + blockSize - 1) / blockSize);

            int* device_idata;
            int* device_odata;

            cudaMalloc((void**)&device_idata, maxSize * sizeof(int));
            cudaMalloc((void**)&device_odata, maxSize * sizeof(int));

            cudaMemset(device_idata, 0, maxSize * sizeof(int));
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO
            for (int d = 0; d <= totalD - 1; d++) {
                kernUpSweep << < fullBlocksPerGrid, blockSize >> > (device_idata, d, maxSize);
            }

            cudaMemset(device_idata + maxSize - 1, 0, sizeof(int));
            for (int d = totalD - 1; d >= 0; d--) {
                kernDownSweep << < fullBlocksPerGrid, blockSize >> > (device_idata, d, maxSize);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, device_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(device_idata);
            cudaFree(device_odata);
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

            int totalD = ilog2ceil(n);
            int maxSize = pow(2, totalD);
            int blockSize = 128;

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 scanBlocksPerGrid((maxSize + blockSize - 1) / blockSize);


            int* device_idata;
            int* device_mappedArr;
            int* device_scannedArr;
            int* device_odata;

            cudaMalloc((void**)&device_idata, maxSize * sizeof(int));
            cudaMalloc((void**)&device_mappedArr, maxSize * sizeof(int));
            cudaMalloc((void**)&device_scannedArr, maxSize * sizeof(int));
            cudaMalloc((void**)&device_odata, maxSize * sizeof(int));

            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          
            timer().startGpuTimer();
            // TODO

            Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (n, device_mappedArr, device_idata);

            cudaMemcpy(device_scannedArr, device_mappedArr, maxSize * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d <= totalD - 1; d++) {
                kernUpSweep << < scanBlocksPerGrid, blockSize >> > (device_scannedArr, d, maxSize);
            }

            cudaMemset(device_scannedArr + maxSize - 1, 0, sizeof(int));
            for (int d = totalD - 1; d >= 0; d--) {
                kernDownSweep << < scanBlocksPerGrid, blockSize >> > (device_scannedArr, d, maxSize);
            }

            //scatter
            Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (maxSize, device_odata, device_idata, device_mappedArr, device_scannedArr);
           
            timer().endGpuTimer();

            int count = 0;
            cudaMemcpy(&count, device_scannedArr + maxSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, device_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(device_idata);
            cudaFree(device_mappedArr);
            cudaFree(device_scannedArr);
            cudaFree(device_odata);

            return count;
        }
    }
}
