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

/**
 * Naive parallel scan algorithm
 * Input must be stored in `data`.
 * Output is stored both in `data` and `buffer`.
 */
__global__ void kernScanInclusiveNaive(int n, int *data, int *buffer) {
  int id    = blockDim.x * blockIdx.x + threadIdx.x;
  int tx    = threadIdx.x;
  int bdim  = blockDim.x;
  int log2n = ilog2ceil((n < bdim) ? n : bdim);

  if (id < n) {
    for (int d = 1; d <= log2n; ++d) {
      buffer[id] = data[id];
      __syncthreads();
      if (tx >= (1 << (d - 1))) {
        buffer[id] = data[id - (1 << (d - 1))] + data[id];
      }
      __syncthreads();
      data[id] = buffer[id];
      __syncthreads();
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

  int *dev_idata, *dev_odata, *buffer;
  cudaMalloc((void **)&dev_idata, n * sizeof(int));
  cudaMalloc((void **)&dev_odata, n * sizeof(int));
  cudaMalloc((void **)&buffer, n * sizeof(int));
  checkCUDAError("cudaMalloc failed for dev_idata, dev_odata, buffer!");

  int *dev_offset_inclusive, *dev_offset_exclusive;
  cudaMalloc((void **)&dev_offset_inclusive, grid_size * sizeof(int));
  cudaMalloc((void **)&dev_offset_exclusive, grid_size * sizeof(int));
  checkCUDAError(
      "cudaMalloc failed for dev_offset_inclusive, dev_offset_exclusive!");

  cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy failed for idata --> dev_idata!");

  /******* KERNEL INVOCATIONS *******/
  timer().startGpuTimer();
  dim3 dimGrid{grid_size}, dimBlock{Common::block_size};
  kernScanInclusiveNaive<<<dimGrid, dimBlock>>>(n, dev_idata, buffer);
  Common::kernExtractLastElementPerBlock<<<dimGrid, dimBlock>>>(
      n, dev_offset_inclusive, dev_idata);
  kernScanInclusiveNaive<<<1, dimBlock>>>(
      grid_size, dev_offset_inclusive,
      dev_offset_exclusive);  // dev_offset_exclusive only serves as buffer here
  Common::kernShiftToExclusive<<<1, dimBlock>>>(grid_size, dev_offset_exclusive,
                                                dev_offset_inclusive);
  Common::kernAddOffsetPerBlock<<<dimGrid, dimBlock>>>(
      n, dev_idata, dev_offset_exclusive, buffer);
  Common::kernShiftToExclusive<<<dimGrid, dimBlock>>>(n, dev_odata, dev_idata);
  timer().endGpuTimer();
  /**********************************/

  cudaDeviceSynchronize();
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_idata);
  cudaFree(dev_odata);
}
}  // namespace Naive
}  // namespace StreamCompaction
