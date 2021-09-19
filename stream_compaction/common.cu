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

const unsigned int block_size = 256;

__global__ void kernExtractLastElementPerBlock(int n, int *odata,
                                               const int *idata) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int id  = bid * blockDim.x + tid;
  if (id < n) {
    if (tid == blockDim.x - 1 || id == n - 1) {
      odata[bid] = idata[id];
    }
  }
}

__global__ void kernAddOffsetPerBlock(int n, int *odata,
                                      const int *block_offset,
                                      const int *idata) {
  int bid = blockIdx.x;
  int id  = bid * blockDim.x + threadIdx.x;
  if (id < n) {
    odata[id] = idata[id] + block_offset[bid];
  }
}

__global__ void kernShiftToExclusive(int n, int *odata, const int *idata) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    if (id == 0)
      odata[id] = 0;
    else
      odata[id] = idata[id - 1];
  }
}

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
  // TODO
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, int *odata, const int *idata,
                            const int *bools, const int *indices) {
  // TODO
}

}  // namespace Common
}  // namespace StreamCompaction
