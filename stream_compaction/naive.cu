#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO:
        
        __global__ void kernNaiveScan(const int n, const int d, int* odata, const int* idata) {

            /*1: for d = 1 to log2 n do
                2 : for all k in parallel do
                3 : if k U2265.GIF 2 d  then
                4 : x[out][k] = x[in][k ¨C 2 d - 1] + x[in][k]
                5 : else
                6 : x[out][k] = x[in][k]*/
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int dPow = powf(2, d - 1);

            if (index >= dPow) {
                odata[index] = idata[index - dPow] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* device_idata;
            int* device_odata;

            cudaMalloc((void**)&device_idata, n * sizeof(int));
            cudaMalloc((void**)&device_odata, n * sizeof(int));

            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan << < fullBlocksPerGrid, blockSize >> > (n, d, device_odata, device_idata);
                int* temp = device_idata;
                device_idata = device_odata;
                device_odata = temp;
            }
            
            timer().endGpuTimer();

            cudaThreadSynchronize();

            cudaMemcpy(odata + 1, device_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(device_idata);
            cudaFree(device_odata);

            
        }
    }
}


//__global__ void scan(float* g_odata, float* g_idata, int n) {
//    extern __shared__ float temp[]; // allocated on invocation    
//    int thid = threadIdx.x;   
//    int pout = 0, pin = 1;   // Load input into shared memory.    
//                             // This is exclusive scan, so shift right by one    
//                             // and set first element to 0   
//    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;   
//    __syncthreads();  
//    for (int offset = 1; offset < n; offset *= 2)   
//    {       
//        pout = 1 - pout; // swap double buffer indices     
//        pin = 1 - pout;     
//        if (thid >= offset)       
//            temp[pout*n+thid] += temp[pin*n+thid - offset];     
//        else       
//            temp[pout*n+thid] = temp[pin*n+thid];     
//        __syncthreads();   
//    }
//    g_odata[thid] = temp[pout*n+thid]; // write output 
//} 