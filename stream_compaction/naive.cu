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
__global__ void kernScanInclusive(int n, int *data, int *buffer) {
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

  int num_scans = 1;
  int len       = n;
  while ((len + Common::block_size - 1) / Common::block_size > 1) {
    ++num_scans;
    len = (len + Common::block_size - 1) / Common::block_size;
  }

  int **dev_idata  = (int **)malloc(num_scans * sizeof(int *));
  int **dev_odata  = (int **)malloc(num_scans * sizeof(int *));
  int **dev_buffer = (int **)malloc(num_scans * sizeof(int *));
  int *array_sizes = (int *)malloc(num_scans * sizeof(int));
  int *grid_sizes  = (int *)malloc(num_scans * sizeof(int));

  len = n;
  for (int i = 0; i < num_scans; ++i) {
    cudaMalloc((void **)&dev_idata[i], len * sizeof(int));
    cudaMalloc((void **)&dev_odata[i], len * sizeof(int));
    cudaMalloc((void **)&dev_buffer[i], len * sizeof(int));
    checkCUDAError("cudaMalloc failed for dev_idata, dev_odata, dev_buffer!");
    array_sizes[i] = len;
    len            = (len + Common::block_size - 1) / Common::block_size;
    grid_sizes[i]  = len;
  }

  cudaMemcpy(dev_idata[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy failed for idata --> dev_idata[0]!");

  /******* KERNEL INVOCATIONS *******/
  dim3 dimBlock{Common::block_size};
  timer().startGpuTimer();
  for (int i = 0; i < num_scans; ++i) {
    dim3 dimGrid{(unsigned int)grid_sizes[i]};
    kernScanInclusive<<<dimGrid, dimBlock>>>(array_sizes[i], dev_idata[i],
                                             dev_buffer[i]);
    if (i < num_scans - 1) {
      Common::kernExtractLastElementPerBlock<<<dimGrid, dimBlock>>>(
          array_sizes[i], dev_idata[i + 1], dev_idata[i]);
    }
  }
  for (int i = num_scans - 1; i >= 0; --i) {
    dim3 dimGrid{(unsigned int)grid_sizes[i]};
    Common::kernShiftToExclusive<<<dimGrid, dimBlock>>>(
        array_sizes[i], dev_odata[i], dev_buffer[i]);
    if (i >= 1) {
      dim3 next_dimGrid{(unsigned int)grid_sizes[i - 1]};
      Common::kernAddOffsetPerBlock<<<next_dimGrid, dimBlock>>>(
          array_sizes[i - 1], dev_buffer[i - 1], dev_odata[i],
          dev_idata[i - 1]);
    }
  }
  cudaDeviceSynchronize();
  timer().endGpuTimer();
  /**********************************/

  cudaMemcpy(odata, dev_odata[0], n * sizeof(int), cudaMemcpyDeviceToHost);

  // Free all memory allocations
  for (int i = 0; i < num_scans; ++i) {
    cudaFree(dev_idata[i]);
    cudaFree(dev_odata[i]);
    cudaFree(dev_buffer[i]);
  }
  free(grid_sizes);
  free(array_sizes);
  free(dev_idata);
  free(dev_odata);
  free(dev_buffer);
}
}  // namespace Naive
}  // namespace StreamCompaction
