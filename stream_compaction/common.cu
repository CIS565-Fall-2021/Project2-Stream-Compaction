#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            // Assume bools initialized with 0s
            if (idata[index]) {
                bools[index] = 1;
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            // Assume bools initialized with 0s
            if (bools[index]) {
                odata[indices[index]] = idata[index];
            }
        }

    }
}

// printf("?????????????????????????????????????????: %d %d %d %d %d %d %d %d\n", odata[0], odata[1], odata[2], odata[3], odata[4], odata[5], odata[6], odata[7]);
// cudaMemcpy(odata, dev_array, sizeof(int) * n, cudaMemcpyDeviceToHost);

//void scan(int n, int *odata, const int *idata) {
//    // Create device pointers
//    int *dev_idata;
//    int *dev_odata;
//    cudaMalloc((void **)&dev_idata, n * sizeof(int));
//    cudaMalloc((void **)&dev_odata, n * sizeof(int));
//    checkCUDAError("cudaMalloc failed!");
//
//    // Copy data to GPU
//    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcpy failed!");
//
//    // Cast to thrust vector
//    thrust::device_vector<int> dev_thrust_idata(dev_idata, dev_idata + n);
//    thrust::device_vector<int> dev_thrust_odata(n);
//
//    timer().startGpuTimer();
//
//    thrust::exclusive_scan(dev_thrust_idata.begin(), dev_thrust_idata.end(), dev_thrust_odata.begin());
//
//    timer().endGpuTimer();
//
//    // Copy data back
//    dev_odata = thrust::raw_pointer_cast(dev_thrust_odata.data());
//    cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudaMemcpy back failed!");
//
//    // Cleanup
//    cudaFree(dev_idata);
//    cudaFree(dev_odata);
//    checkCUDAError("cudaFree failed!");
//}