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

        int* dev_buf;
        int* dev_bools;
        int* dev_idata;
        int* dev_odata;
        int* dev_scanned;

        __global__ void kernUpSweep(int N, int* data, int offset) {
            int idx = threadIdx.x + (blockDim.x * blockIdx.x);
            if (idx >= N) {
                return;
            }
            
            if (idx % (2 * offset) == 0) {
                data[idx + offset * 2 - 1] += data[idx + offset - 1];
            }
        }

        __global__ void kernDownSweep(int N, int* data, int offset) {
            int idx = threadIdx.x + (blockDim.x * blockIdx.x);
            if (idx >= N) {
                return;
            }

            if (idx % (2 * offset) == 0) {
                int temp = data[idx + offset - 1];
                data[idx + offset - 1] = data[idx + offset * 2 - 1];
                data[idx + offset * 2 - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO (DONE)

            //other variables
            int blockSize = 128;
            int max = ilog2ceil(n);
            int numObj = (int)powf(2, max);
            dim3 numBlocks((numObj + blockSize - 1) / blockSize);

            //malloc + memcopy
            cudaMalloc((void**)&dev_buf, sizeof(int) * numObj);
            cudaMemcpy(dev_buf, idata, numObj * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            //call kernels
            for (int i = 0; i < max; i++) {
                kernUpSweep<<<numBlocks, blockSize>>>(numObj, dev_buf, (int)powf(2, i));
            }
            cudaMemset(dev_buf + numObj - 1, 0, sizeof(int));
            for (int i = max - 1; i >= 0; i--) {
                kernDownSweep<<<numBlocks, blockSize>>>(numObj, dev_buf, (int)powf(2, i));
            }           
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_buf);
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
            //variables
            int blockSize = 128;
            int max = ilog2ceil(n);
            int numObj = (int)powf(2, max);
            dim3 numBlocks((numObj + blockSize - 1) / blockSize);

            //malloc
            cudaMalloc((void**)&dev_bools, sizeof(int) * numObj);
            cudaMalloc((void**)&dev_idata, sizeof(int) * numObj);
            cudaMalloc((void**)&dev_odata, sizeof(int) * numObj);
            cudaMalloc((void**)&dev_scanned, sizeof(int) * numObj);

            cudaMemcpy(dev_idata, idata, numObj * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO (DONE)
            StreamCompaction::Common::kernMapToBoolean<<<numBlocks, blockSize>>>(numObj, dev_bools, dev_idata);
            cudaMemcpy(dev_scanned, dev_bools, sizeof(int) * numObj, cudaMemcpyDeviceToDevice);
            
            for (int i = 0; i < max; i++) {
                kernUpSweep<<<numBlocks, blockSize>>>(numObj, dev_scanned, (int)powf(2, i));
            }
            cudaMemset(dev_scanned + numObj - 1, 0, sizeof(int));
            for (int i = max - 1; i >= 0; i--) {
                kernDownSweep<<<numBlocks, blockSize>>>(numObj, dev_scanned, (int)powf(2, i));
            }

            StreamCompaction::Common::kernScatter<<<numBlocks, blockSize>>>(numObj, dev_odata, dev_idata, dev_bools, dev_scanned);

            timer().endGpuTimer();

            int* arr = new int[numObj];
            cudaMemcpy(arr, dev_bools, sizeof(int) * numObj, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_scanned);
            cudaFree(dev_odata);

            int count = 0;
            for (int i = 0; i < n; i++) {
                if (arr[i] == 1) {
                    count++;
                }
            }          
            return count;
        }
    }
}
