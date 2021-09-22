#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

#define checkCUDAErrorWithLine(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)
#define blockSize 512


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int N, int d, int* odata, const int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N) return;

//          int power = 1;
//          if(d>1){
//            for(int i=0; i<d-1; i++){
//              power *= 2;
//            }
//          }
          int power = (int) powf(2.0f, (float) d-1);


          if (index >= power){

            odata[index] = idata[index-power] + idata[index];


          }else{
            odata[index] = idata[index];
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
        void scan(int _N, int *odata, const int *idata) {

          int N = getNextPower(_N);

          dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

          // copy data to gpu buffer
          int* dev_odata;
          int* dev_idata;
          cudaMalloc((void**)&dev_odata, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
          cudaMalloc((void**)&dev_idata, N * sizeof(int));
          checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

          cudaMemcpy(dev_idata, idata, sizeof(int) * _N, cudaMemcpyHostToDevice);
          cudaMemcpy(dev_odata, idata, sizeof(int) * _N, cudaMemcpyHostToDevice);


          timer().startGpuTimer();

          for(int d=1;d<=ilog2ceil(N);d++){
            int* tmp = dev_idata;
            dev_idata = dev_odata;
            dev_odata = tmp;

            kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_odata, dev_idata);

            cudaDeviceSynchronize();

          }

          cudaDeviceSynchronize();
          timer().endGpuTimer();

          cudaMemcpy(odata, dev_odata, sizeof(int) * _N, cudaMemcpyDeviceToHost);

          cudaDeviceSynchronize();
          cudaFree(dev_odata);
          cudaFree(dev_idata);
        }
    }
}
