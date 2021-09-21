#include <cstdio>

#include "common.h"
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

/**
 * CPU scan core function.
 * This function runs without starting CPU timer.
 */
void scan_core(int n, int *odata, const int *idata) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    odata[i] = sum;
    sum += idata[i];
  }
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can
 * simulate your GPU scan in this function first.
 */
void scan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  scan_core(n, odata, idata);
  timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  int outarray_len = 0;
  for (int i = 0; i < n; ++i) {
    if (idata[i] != 0) {
      odata[outarray_len++] = idata[i];
    }
  }
  timer().endCpuTimer();
  return outarray_len;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();

  // create bool array b
  int *b = (int *)malloc(n * sizeof(int));
  std::memset(b, 0, n * sizeof(int));
  for (int i = 0; i < n; ++i) {
    if (idata[i] != 0) b[i] = 1;
  }

  // exclusive scan bool array
  int *scan_b = (int *)malloc(n * sizeof(int));
  scan_core(n, scan_b, b);
  int outarray_len = b[n - 1] + scan_b[n - 1];

  // copy selected array into out array
  for (int i = 0; i < n; ++i) {
    if (b[i]) odata[scan_b[i]] = idata[i];
  }

  free(b);
  free(scan_b);

  timer().endCpuTimer();
  return outarray_len;
}
}  // namespace CPU
}  // namespace StreamCompaction
