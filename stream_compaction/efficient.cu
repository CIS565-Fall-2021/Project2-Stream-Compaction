#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define checkCUDAErrorWithLine(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)
#define blockSize 64

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEfficientScanUp(int N, int d, int* odata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N) return;

          int power_plus = (int) powf(2.0f, (float) d+1);
          if (index % power_plus != 0) return;

          int power_ident = power_plus / 2;
          //int power_ident = (int) powf(2.0f, (float) d);

          odata[index + power_plus - 1] += odata[index + power_ident - 1];


        }

        __global__ void kernEfficientScanReset(int N, int* odata) {
          // should not be needed in the end
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index != 0) return;
          odata[N-1] = 0;
        }




        __global__ void kernEfficientMoveLastValue(int N, int block_idx, const int* odata, int* auxArray) {
          // should not be needed in the end
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index != 0) return;
          auxArray[block_idx] = odata[N-1];


        }

        __global__ void kernEfficientScanDown(int N, int d, int* odata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N) return;

          int power_plus = (int) powf(2.0f, (float) d+1);
          if (index % power_plus != 0) return;

          int power_ident = power_plus / 2;
          //int power_ident = (int) powf(2.0f, (float) d);

          int t = odata[index + power_ident - 1];

          odata[index + power_ident - 1] = odata[index + power_plus - 1];
          odata[index + power_plus - 1] += t;


        }

        __global__ void kernAddToAll(int N, int block_idx, const int* auxArr, int* odata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N) return;

          odata[index] += auxArr[block_idx];
        }

        void _scan(int N, int* dev_odata) {
          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);



          for(int d=0;d<=ilog2ceil(N)-1;d++){
            kernEfficientScanUp<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_odata);

          }

          kernEfficientScanReset<<<fullBlocksPerGrid, blockSize>>>(N, dev_odata);

          for(int d=ilog2ceil(N)-1;d>=0;d--){
            kernEfficientScanDown<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_odata);


          }


        }

        void _scanBlock(int N, int* dev_odata, int block_idx, int* aux_arr) {
          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          int offset = N*block_idx;

          for(int d=0;d<=ilog2ceil(N)-1;d++){
            kernEfficientScanUp<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_odata + offset);

          }

          cudaMemcpy(aux_arr + block_idx, dev_odata + offset + N - 1, sizeof(int), cudaMemcpyDeviceToDevice);
          //cudaMemset(dev_odata + offset + N -1 , 0, sizeof(int));
          kernEfficientScanReset<<<fullBlocksPerGrid, blockSize>>>(N, dev_odata + offset);

          for(int d=ilog2ceil(N)-1;d>=0;d--){
            kernEfficientScanDown<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_odata + offset);


          }


        }


        int getNextPower(int _N){
          int N = 1;
          while(N < _N){
            N *= 2;
          }
          return N;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */


        void _chunkedScanAutomatic(int N, int *odata){


          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          // divide array into blocks
          const int num_per_block = 1024;
          int num_blocks = ceil(N / (double) num_per_block);
          int num_scan_blocks = getNextPower(num_blocks);

          int* dev_auxarr;
          cudaMalloc((void**)&dev_auxarr, num_scan_blocks * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_auxarr failed!");


          for(int block_idx=0; block_idx<num_blocks; block_idx++){
            _scanBlock(num_per_block, odata, block_idx, dev_auxarr);
          }
          _scan(num_scan_blocks, dev_auxarr);

          for(int block_idx=1; block_idx<num_blocks; block_idx++){
            int offset = num_per_block*block_idx;
            kernAddToAll<<<fullBlocksPerGrid, blockSize>>>(num_per_block, block_idx, dev_auxarr, odata + offset);
          }

          cudaFree(dev_auxarr);

        }

        void _chunkedScan(int N, int *odata, int *auxarr, int num_per_block, int num_blocks, int num_scan_blocks){


          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);






          for(int block_idx=0; block_idx<num_blocks; block_idx++){
            _scanBlock(num_per_block, odata, block_idx, auxarr);
          }
          _scan(num_scan_blocks, auxarr);

          for(int block_idx=1; block_idx<num_blocks; block_idx++){
            int offset = num_per_block*block_idx;
            kernAddToAll<<<fullBlocksPerGrid, blockSize>>>(num_per_block, block_idx, auxarr, odata + offset);
          }



        }


        void scan(int _N, int *odata, const int *idata) {


          int N = getNextPower(_N);


          //dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          // divide array into blocks
          const int num_per_block = 1024;
          int num_blocks = ceil(N / (double) num_per_block);
          int num_scan_blocks = getNextPower(num_blocks);

          int* dev_auxarr;
          cudaMalloc((void**)&dev_auxarr, num_scan_blocks * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_auxarr failed!");

          // copy data to gpu buffer
          int* dev_odata;
          cudaMalloc((void**)&dev_odata, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

          cudaMemset(dev_odata, 0, sizeof(int) * N);
          cudaMemcpy(dev_odata, idata, sizeof(int) * _N, cudaMemcpyHostToDevice);

          timer().startGpuTimer();

          _chunkedScan(N, dev_odata, dev_auxarr, num_per_block, num_blocks, num_scan_blocks);


          timer().endGpuTimer();

          cudaMemcpy(odata, dev_odata, sizeof(int) * _N, cudaMemcpyDeviceToHost);

          cudaDeviceSynchronize();
          cudaFree(dev_odata);
          cudaFree(dev_auxarr);

        }

        void scanplain(int _N, int *odata, const int *idata) {


          int N = getNextPower(_N);

          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);


          // copy data to gpu buffer
          int* dev_odata;
          //int* dev_idata;
          cudaMalloc((void**)&dev_odata, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

          cudaMemset(dev_odata, 0, sizeof(int) * N);
          cudaMemcpy(dev_odata, idata, sizeof(int) * _N, cudaMemcpyHostToDevice);

          timer().startGpuTimer();

          _scan(N, dev_odata);

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_odata, sizeof(int) * _N, cudaMemcpyDeviceToHost);

          cudaDeviceSynchronize();
          cudaFree(dev_odata);
        }


        __global__ void kernBooleanAssigner(int N, int* odata, const int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N) return;

          odata[index] = idata[index] == 0 ? 0 : 1;

        }

        __global__ void kernScatter(int N, int* odata, const int* idata, const int* boolarr, const int* scanres ) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N) return;

          if(!boolarr[index]) return;

          odata[scanres[index]] = idata[index];


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
        int compact(int _N, int *odata, const int *idata) {

          int N = getNextPower(_N);

          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          // copy data to gpu buffer
          int* dev_odata;
          int* dev_idata;
          int* dev_1bool;
          int* dev_2scanres;
          int numElements = -1;


          cudaMalloc((void**)&dev_odata, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
          cudaMalloc((void**)&dev_idata, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
          cudaMalloc((void**)&dev_1bool, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_1bool failed!");
          cudaMalloc((void**)&dev_2scanres, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_2scanres failed!");

          cudaMemset(dev_odata, 0, sizeof(int) * N);
          cudaMemset(dev_idata, 0, sizeof(int) * N);
          cudaMemset(dev_1bool, 0, sizeof(int) * N);
          cudaMemset(dev_2scanres, 0, sizeof(int) * N);

          cudaMemcpy(dev_idata, idata, sizeof(int) * _N, cudaMemcpyHostToDevice);
          //cudaMemcpy(dev_odata, odata, sizeof(int) * N, cudaMemcpyHostToDevice);

          timer().startGpuTimer();

          kernBooleanAssigner<<<fullBlocksPerGrid, blockSize>>>(N, dev_1bool, dev_idata);
          cudaMemcpy(dev_2scanres, dev_1bool, sizeof(int) * _N, cudaMemcpyDeviceToDevice);


          _chunkedScanAutomatic(_N, dev_2scanres);


          kernScatter<<<fullBlocksPerGrid, blockSize>>>(N, dev_odata, dev_idata, dev_1bool, dev_2scanres);
          cudaMemcpy(&numElements, dev_2scanres + (_N-1), sizeof(int), cudaMemcpyDeviceToHost);

          if(_N % 2 != 0){
            numElements++;
          }

          cudaDeviceSynchronize();
          timer().endGpuTimer();

          cudaMemcpy(odata, dev_odata, sizeof(int) * _N, cudaMemcpyDeviceToHost);

          cudaDeviceSynchronize();
          cudaFree(dev_odata);
          cudaFree(dev_idata);
          cudaFree(dev_1bool);
          cudaFree(dev_2scanres);


            return numElements;
        }
    }
}
