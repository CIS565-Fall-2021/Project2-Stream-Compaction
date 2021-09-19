#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

__global__ void kernScanInclusive(int n, int *odata, int *idata) {
  int tid   = threadIdx.x;
  int bdim  = blockDim.x;
  int id    = blockIdx.x * bdim + tid;
  int log2n = ilog2ceil((n < bdim) ? n : bdim);
  if (id < n) {
    // upsweep
    for (int d = 0; d < log2n; ++d) {
      if (id % (1 << (d + 1)) == 0) {
        idata[id + (1 << (d + 1)) - 1] += idata[id + (1 << d) - 1];
      }
      __syncthreads();
    }

    // last thread remembers and sets reduction sum after downsweep
    int reduction_sum = 0;
    if (tid == bdim - 1 || id == n - 1) {
      reduction_sum = idata[id];
      idata[id]     = 0;
    }
    __syncthreads();

    // downsweep
    for (int d = log2n - 1; d >= 0; --d) {
      if (id % (1 << (d + 1)) == 0) {
        int temp                 = idata[id + (1 << d) - 1];
        idata[id + (1 << d) - 1] = idata[id + (1 << (d + 1)) - 1];
        idata[id + (1 << (d + 1)) - 1] += temp;
      }
      __syncthreads();
    }

    // turn exclusive scan into inclusive scan
    if (tid == bdim - 1 || id == n - 1) {
      odata[id] = reduction_sum;
    } else {
      odata[id] = idata[id + 1];
    }
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  const unsigned int grid_size =
      (n + Common::block_size - 1) / Common::block_size;
  const int n_pad = 1 << (ilog2ceil(n));

  // allocate input/output device data
  int *dev_idata, *dev_odata;
  cudaMalloc((void **)&dev_idata, n_pad * sizeof(int));
  cudaMalloc((void **)&dev_odata, n_pad * sizeof(int));
  checkCUDAError("cudaMalloc dev_idata, dev_odata failed!");

  cudaMemset(dev_idata, 0, n_pad * sizeof(int));
  checkCUDAError("cudaMemset dev_idata to 0 failed!");
  cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy dev_idata from idata failed!");

  // allocate block reduction device data
  int *dev_block_offset_inclusive, *dev_block_offset_exclusive;
  cudaMalloc((void **)&dev_block_offset_inclusive, grid_size * sizeof(int));
  cudaMalloc((void **)&dev_block_offset_exclusive, grid_size * sizeof(int));

  /******* KERNEL INVOCATION *******/
  dim3 dimGrid{grid_size}, dimBlock{Common::block_size};
  timer().startGpuTimer();
  kernScanInclusive<<<dimGrid, dimBlock>>>(n_pad, dev_odata, dev_idata);
  Common::kernExtractLastElementPerBlock<<<dimGrid, dimBlock>>>(
      n_pad, dev_block_offset_exclusive, dev_odata);
  kernScanInclusive<<<1, dimBlock>>>(grid_size, dev_block_offset_inclusive,
                                     dev_block_offset_exclusive);
  Common::kernShiftToExclusive<<<1, dimBlock>>>(
      grid_size, dev_block_offset_exclusive, dev_block_offset_inclusive);
  Common::kernAddOffsetPerBlock<<<dimGrid, dimBlock>>>(
      n_pad, dev_idata, dev_block_offset_exclusive, dev_odata);
  Common::kernShiftToExclusive<<<dimGrid, dimBlock>>>(n_pad, dev_odata,
                                                      dev_idata);
  timer().endGpuTimer();
  /*********************************/

  cudaDeviceSynchronize();
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy odata from dev_idata failed!");
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
  const unsigned int grid_size =
      (n + Common::block_size - 1) / Common::block_size;
  const int n_pad = 1 << (ilog2ceil(n));

  // Device data allocation
  int *dev_idata, *dev_odata;
  cudaMalloc((void **)&dev_idata, n_pad * sizeof(int));
  cudaMalloc((void **)&dev_odata, n_pad * sizeof(int));
  checkCUDAError("cudaMalloc dev_idata, dev_odata failed!");
  cudaMemset(dev_idata, 0, n_pad * sizeof(int));
  checkCUDAError("cudaMemset dev_idata to 0 failed!");

  cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy dev_idata from idata failed!");

  // helper data allocation
  int *dev_indices, *dev_bools, *dev_buffer;
  cudaMalloc((void **)&dev_indices, n_pad * sizeof(int));
  cudaMalloc((void **)&dev_bools, n_pad * sizeof(int));
  cudaMalloc((void **)&dev_buffer, n_pad * sizeof(int));
  checkCUDAError("cudaMalloc dev_indices, dev_bools, dev_buffer failed!");

  int *dev_offset_inclusive, *dev_offset_exclusive;
  cudaMalloc((void **)&dev_offset_exclusive, grid_size * sizeof(int));
  cudaMalloc((void **)&dev_offset_inclusive, grid_size * sizeof(int));
  checkCUDAError(
      "cudaMalloc dev_offset_inclusive, dev_offset_exclusive failed!");

  /******* KERNEL INVOCATION *******/
  dim3 dimGrid{grid_size}, dimBlock{Common::block_size};
  timer().startGpuTimer();
  Common::kernMapToBoolean<<<dimGrid, dimBlock>>>(n_pad, dev_bools, dev_idata);
  cudaDeviceSynchronize();
  cudaMemcpy(dev_buffer, dev_bools, n_pad * sizeof(int),
             cudaMemcpyDeviceToDevice);
  kernScanInclusive<<<dimGrid, dimBlock>>>(n_pad, dev_odata, dev_buffer);
  Common::kernExtractLastElementPerBlock<<<dimGrid, dimBlock>>>(
      n_pad, dev_offset_exclusive, dev_odata);
  kernScanInclusive<<<1, dimBlock>>>(grid_size, dev_offset_inclusive,
                                     dev_offset_exclusive);
  Common::kernShiftToExclusive<<<1, dimBlock>>>(grid_size, dev_offset_exclusive,
                                                dev_offset_inclusive);
  Common::kernAddOffsetPerBlock<<<dimGrid, dimBlock>>>(
      n_pad, dev_buffer, dev_offset_exclusive, dev_odata);
  Common::kernShiftToExclusive<<<dimGrid, dimBlock>>>(n_pad, dev_indices,
                                                      dev_buffer);
  Common::kernScatter<<<dimGrid, dimBlock>>>(n_pad, dev_odata, dev_idata,
                                             dev_bools, dev_indices);
  timer().endGpuTimer();
  /*********************************/

  // transfer output data to CPU & analyze
  cudaDeviceSynchronize();
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy odata from dev_odata failed!");

  // calculate num. of elements after compaction
  int *indices = (int *)malloc(n * sizeof(int));
  int *bools   = (int *)malloc(n * sizeof(int));
  cudaMemcpy(indices, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError(
      "cudaMemcpy indices from dev_indices, bools from dev_bools failed!");
  int compact_len = indices[n - 1] + bools[n - 1];
  free(indices);
  free(bools);

  return compact_len;
}
}  // namespace Efficient
}  // namespace StreamCompaction
