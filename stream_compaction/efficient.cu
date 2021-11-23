#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "cVec.h"

#define blockSize 32

namespace StreamCompaction {
namespace Efficient {
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}


	__global__ void kern_up_sweep(int d, int n, int *x)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		int k = idx * (1 << (d+1));
		if (k >= n)
			return;
		x[k + (1<<(d+1)) - 1] += x[k + (1<<d) - 1];
	}

	__global__ void kern_down_sweep(int d, int n, int *x)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		int k = idx * (1 << (d+1));

		int t = x[k+(1<<d)-1];
		x[k+(1<<d)-1] = x[k+(1<<(d+1))-1];
		x[k+(1<<(d+1))-1] += t;
	}


	/* in-place scan over device array, doesn't start GPU Timer and assumes input is power of 2 */
	void scan_dev(int N, cVec<int> *dev_data) {
		int log2n = ilog2ceil(N);
		//printf("N: %d\tlog2n: %d\n", N, log2n);
		for (int d = 0; d < log2n; d++) {
			int count = N / (1 << d);
			dim3 fullBlocksPerGrid((count + blockSize - 1) / blockSize);
			kern_up_sweep<<<fullBlocksPerGrid, blockSize>>>(d, N, dev_data->raw_data());
		}
		dev_data->memset(N-1, 1, 0);
		for (int d = log2n - 1; d >= 0; d--) {
			int count = N / (1 << d);
			dim3 fullBlocksPerGrid((count + blockSize - 1) / blockSize);
			kern_down_sweep<<<fullBlocksPerGrid, blockSize>>>(d, N, dev_data->raw_data());
		}
	}

	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata) {
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		cVec<int> dev_data(N, n, idata);

		timer().startGpuTimer();

		scan_dev(N, &dev_data);

		timer().endGpuTimer();

		dev_data.copy_to_host(0, n, odata);
		printf("here\n");
	}


	__global__ void kern_bmap(int n, const int *__restrict__ idata, int *__restrict__ bdata)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		
		if (idx >= n)
			return;

		bdata[idx] = bool(idata[idx]);
	}


	__global__ void kern_scatter(int n, const int *__restrict__ idata, const int *__restrict__ bdata,
					const int *__restrict__ sdata, int *__restrict odata)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (idx >= n)
			return;

		if (bdata[idx])
			odata[sdata[idx]] = idata[idx];
	}


	/**
	 * Performs stream compaction on idata, storing the result into odata.
	 * All zeroes are discarded.
	 *
	 * @param n      The number of elements in idata.
	 * @param odata  The array into which to store elements.
	 * @param idata  The array of elements to compact.
	 * @returns	The number of elements remaining after compaction.
	 */
	int compact(int n, int *odata, const int *idata) {
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		cVec<int> dev_idata(n, n, idata), dev_bdata(n), dev_sdata(N), dev_odata(n);

		timer().startGpuTimer();
		
		/* map to bools */
		printf("start\n");
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		kern_bmap<<<fullBlocksPerGrid, blockSize>>>(n, dev_idata.raw_data(), dev_bdata.raw_data());

		printf("bool\n");
		dev_sdata.copy_to_range(0, n, dev_bdata);
		scan_dev(N, &dev_sdata);

		printf("compact_scan\n");
		kern_scatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_idata.raw_data(), dev_bdata.raw_data(),
			dev_sdata.raw_data(), dev_odata.raw_data());

		timer().endGpuTimer();


		dev_odata.copy_to_host(0, n, odata);

		for (int i = 0; i < n; i++)
			if (!odata[i])
				return i;
		
		return n;
	}
}
}
