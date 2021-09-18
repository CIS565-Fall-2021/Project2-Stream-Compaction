#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

template <typename T>
__device__ void inline swap(T &a, T &b) {
  T c(a);
  a = b;
  b = c;
}

__global__ void kernScanExclusiveNaive(int n, int *idata, int *odata) {
  int id    = blockDim.x * blockIdx.x + threadIdx.x;
  int tx    = threadIdx.x;
  int bdim  = blockDim.x;
  int log2n = ilog2ceil((n < bdim) ? n : bdim);

  if (id < n) {
    for (int d = 1; d <= log2n; ++d) {
      odata[id] = idata[id];
      __syncthreads();
      if (tx >= (1 << (d - 1))) {
        odata[id] = idata[id - (1 << (d - 1))] + idata[id];
      }
      __syncthreads();
      swap(idata, odata);
      __syncthreads();
    }

    if (tx > 0) {
      odata[id] = idata[id - 1];
    } else {
      odata[id] = 0;
    }
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  if (n <= 0) return;

  const unsigned int grid_size =
      (n + Common::block_size - 1) / Common::block_size;

  int *dev_idata, *dev_odata;
  cudaMalloc((void **)&dev_idata, n * sizeof(int));
  cudaMalloc((void **)&dev_odata, n * sizeof(int));
  checkCUDAError("cudaMalloc failed for dev_idata and dev_odata  !");

  cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy failed for idata --> dev_idata!");

  timer().startGpuTimer();
  dim3 dimGrid{grid_size}, dimBlock{Common::block_size};
  kernScanExclusiveNaive<<<dimGrid, dimBlock>>>(n, dev_idata, dev_odata);
  timer().endGpuTimer();

  cudaDeviceSynchronize();
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_idata);
  cudaFree(dev_odata);
}
}  // namespace Naive
}  // namespace StreamCompaction
