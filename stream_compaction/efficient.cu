#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "efficient.h"

#define blockSize 128
#define B 128  // Number of elements processed per block

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int d, int* data) {
          //parallel reduction
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          int k = index * 2 * d;
          int r = k + 2 * d - 1;
          int l = k + d - 1;

          if (r < N && r >= 0) {
            // right node += left node
            data[r] += data[l];
          }
        }

        __global__ void kernUpdateElement(int i, int* arr, int val) {
          arr[i] = val;
        }

        __global__ void kernDownSweep(int N, int d, int* data) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          int k = index * 2 * d;
          int r = k + 2 * d - 1;
          int l = k + d - 1;

          if (r < N && r >= 0) {
            int t = data[l];  //left child
            data[l] = data[r];  //left child = right child (current)
            data[r] += t;  // right child += previous left child
          }
        }

        __global__ void kernMakeExclusive(int N, int* data, int auxOffset) {

          extern __shared__ int tmp[];

          int tid = threadIdx.x;
          int index = threadIdx.x + (blockDim.x * blockIdx.x);
          if (index >= N)
            return;

          tmp[tid] = data[auxOffset + index - 1];

          __syncthreads();

          data[auxOffset + index] = (index > 0) ? tmp[tid] : 0;
        }

        __global__ void kernMakeInclusive(int N, int* odata, int* scan_data, int* idata) {

          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= N) {
            return;
          }

          if (index == N - 1) {
            odata[index] = scan_data[index] + idata[index];
          }
          else {
            odata[index] = scan_data[index + 1];
          }
        }

        __global__ void kernAddBlockSum(int N, int* data, int* aux, int dataOffset, int auxOffset) {
          int index = threadIdx.x + blockDim.x * blockIdx.x;
          if (index < N)
            data[dataOffset + index] += aux[auxOffset + blockIdx.x];
        }

        /**
         * Performs the full scan inside a kernel instead of breaking it apart
         */
        __global__ void kernScan(int N, int *data, int *aux, int dataOffset, int auxOffset) {

          extern __shared__ int temp[];  // dynamic shared array

          int tid = threadIdx.x;  // [0 - B/2)
          int idx = tid + blockDim.x * blockIdx.x;
          int n = blockDim.x * 2;  // B
          
          // NOTE: Is it fine to read and write to data arr in the same kernel?
          temp[2 * tid] = data[dataOffset + 2 * idx];
          temp[2 * tid + 1] = data[dataOffset + 2 * idx + 1];

          for (int d = 1; d < n; d <<= 1) {

            __syncthreads();

            int k = tid * 2 * d;
            int r = k + 2 * d - 1;
            int l = k + d - 1;
            if (r < n) {
              temp[r] += temp[l];
            }
          }

          __syncthreads();
          if (tid == blockDim.x - 1)
            aux[auxOffset + blockIdx.x] = temp[n - 1];
         
          __syncthreads();
          if (tid == 0)
            temp[n - 1] = 0;  // set root to 0

          for (int d = n >> 1; d > 0; d >>= 1) {

            __syncthreads();

            int k = tid * 2 * d;
            int r = k + 2 * d - 1;
            int l = k + d - 1;
            if (r < n) {
              int t = temp[l];    // left child
              temp[l] = temp[r];  // left child = right child (current)
              temp[r] += t;       // right child += previous left child
            }
          }

          __syncthreads();

          // make inclusive
          if (tid == blockDim.x - 1) {
            data[dataOffset + 2 * idx] = temp[2 * tid + 1];
            data[dataOffset + 2 * idx + 1] += temp[2 * tid + 1];
          }
          else {
            data[dataOffset + 2 * idx] = temp[2 * tid + 1];
            data[dataOffset + 2 * idx + 1] = temp[2 * tid + 2];
          }
        }

        void recursiveScan(int n, int* dev_data, int* dev_aux, int dataOffset, int auxOffset, int *odata) {
          int M = std::max(n / B, 1);
          kernScan<<<M, B / 2, sizeof(int) * B>>> (M, dev_data, dev_aux, dataOffset, auxOffset);
          checkCUDAErrorFn("kernScan Aux failed.");

          if (M > 1) {
            recursiveScan(M, dev_aux, dev_aux, auxOffset, auxOffset + M, odata);
          }

          int m = std::max(M / B, 1);
          kernMakeExclusive<<<m, B, sizeof(int) * B>>>(M, dev_aux, auxOffset);
          checkCUDAErrorFn("kernMakeExclusive Aux failed.");

          kernAddBlockSum<<<M, B>>>(n, dev_data, dev_aux, dataOffset, auxOffset);
          checkCUDAErrorFn("kernAddBlockSum failed.");
        }

        void recursiveScan(int n, int* odata, const int* idata) {
          int N = imakepower2(n);

          int* dev_data;
          int* dev_aux;

          cudaMalloc((void**)&dev_data, sizeof(int) * N);
          checkCUDAErrorFn("dev_data malloc failed.");
          cudaMalloc((void**)&dev_aux, sizeof(int) * N);  // allocate maximum aux size 
          checkCUDAErrorFn("dev_aux malloc failed.");

          cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          checkCUDAErrorFn("memCpy from idata to dev_data failed.");

          cudaMemset(dev_aux, 0, sizeof(int) * N);
          checkCUDAErrorFn("aux memset failed.");

          timer().startGpuTimer();

          recursiveScan(N, dev_data, dev_aux, 0, 0, odata);

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
          checkCUDAErrorFn("memCpy from dev_data to odata failed.");

          cudaFree(dev_data);
          cudaFree(dev_aux);
          checkCUDAErrorFn("cudaFree failed.");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
          int N = imakepower2(n);

          int* dev_data;
          int* dev_data2;
          int* dev_aux;

          cudaMalloc((void**)&dev_data, sizeof(int) * N);
          checkCUDAErrorFn("dev_data malloc failed.");
          cudaMalloc((void**)&dev_data2, sizeof(int) * N);
          checkCUDAErrorFn("dev_data malloc failed.");

          cudaMalloc((void**)&dev_aux, sizeof(int) * N);  // allocate maximum aux size 
          checkCUDAErrorFn("dev_aux malloc failed.");

          cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          checkCUDAErrorFn("memCpy from idata to dev_data failed.");
          cudaMemcpy(dev_data2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          checkCUDAErrorFn("memCpy from idata to dev_data2 failed.");
          
          cudaMemset(dev_aux, 0, sizeof(int) * N);
          checkCUDAErrorFn("aux memset failed.");

          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          int T = blockSize;
          int M = N / T;

          timer().startGpuTimer();

          for (int d = 1; d < N; d <<= 1) {
            while (T * d > N && T > 1)
              T >>= 1;  // shrink block size
            kernUpSweep<<<M, T>>>(N, d, dev_data);
            M = std::max(M >> 1, 1);  // divide by 2
          }
          //cudaDeviceSynchronize();
          checkCUDAErrorFn("kernUpSweep failed.");

          // set last element to 0
          cudaMemset(dev_data + N - 1, 0, sizeof(int));
          checkCUDAErrorFn("memset failed.");

          M = 1;
          for (int d = N >> 1; d > 0; d >>= 1) {
            while (T * d < N && T < blockSize)
              T <<= 1;  // expand block size
            kernDownSweep<<<M, T>>>(N, d, dev_data);
            M = std::min(M << 1, N / T);
          }
          //cudaDeviceSynchronize();
          checkCUDAErrorFn("kernDownSweep failed.");

          kernMakeInclusive<<<fullBlocksPerGrid, blockSize>>>(N, dev_data2, dev_data, dev_data2);
          checkCUDAErrorFn("kernMakeInclusive failed.");

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_data2, sizeof(int) * n, cudaMemcpyDeviceToHost);
          checkCUDAErrorFn("memCpy from dev_data to odata failed.");

          cudaFree(dev_data);
          cudaFree(dev_data2);
          cudaFree(dev_aux);
          checkCUDAErrorFn("cudaFree failed.");
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
            int N = imakepower2(n);

            int* dev_idata;
            int* dev_indices;
            int* dev_bools;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, sizeof(int) * N);
            checkCUDAErrorFn("dev_idata malloc failed.");
            cudaMalloc((void**)&dev_indices, sizeof(int) * N);
            checkCUDAErrorFn("dev_indices malloc failed.");
            cudaMalloc((void**)&dev_bools, sizeof(int) * N);
            checkCUDAErrorFn("dev_bools malloc failed.");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
          
            int T = blockSize;
            int M = N / T;

            timer().startGpuTimer();
            
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAErrorFn("kernMapToBoolean failed.");
            
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            for (int d = 1; d < N; d <<= 1) {
              while (T * d > N && T > 1)
                T >>= 1;  // shrink block size
              kernUpSweep<<<M, T>>>(N, d, dev_indices);
              M = std::max(M >> 1, 1);  // divide by 2
            }
            checkCUDAErrorFn("kernUpSweep failed.");

            // grab the size
            int L;
            cudaMemcpy(&L, &dev_indices[N-1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcpy dev_indices failed.");
            cudaMalloc((void**)&dev_odata, sizeof(int) * L);

            cudaMemset(&dev_indices[N - 1], 0, sizeof(int));

            M = 1;
            for (int d = N >> 1; d > 0; d >>= 1) {
              while (T * d < N && T < blockSize)
                T <<= 1;  // expand block size
              kernDownSweep<<<M, T>>>(N, d, dev_indices);
              M = std::min(M << 1, N / T);
            }
            checkCUDAErrorFn("kernDownSweep failed.");

            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(N, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAErrorFn("kernScatter failed.");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * L, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memCpy from dev_data to odata failed.");

            cudaFree(dev_bools);
            cudaFree(dev_idata);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return L;
        }
    }
}
