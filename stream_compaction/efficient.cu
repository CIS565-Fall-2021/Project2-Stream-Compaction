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
	void scan_dev(int N, cu::cVec<int> *dev_data) {
		int log2n = ilog2ceil(N);
		int fullBlocksPerGrid = (N + blockSize - 1)/blockSize;
		for (int d = 0; d < log2n; d++) {
			//int count = N / (1 << d);
//				dim3 fullBlocksPerGrid((count + blockSize - 1) / blockSize);
			kern_up_sweep<<<fullBlocksPerGrid, blockSize>>>(d, N, dev_data->raw_ptr());
		}
		cu::memset(dev_data->ptr() + N-1, 0, 1);
		for (int d = log2n - 1; d >= 0; d--) {
		//	int count = N / (1 << d);
		//	dim3 fullBlocksPerGrid((count + blockSize - 1) / blockSize);
			kern_down_sweep<<<fullBlocksPerGrid, blockSize>>>(d, N, dev_data->raw_ptr());
		}
	}

	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata) {
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		cu::cVec<int> dev_data(n, idata, N);

		timer().startGpuTimer();

		scan_dev(N, &dev_data);

		timer().endGpuTimer();

		cu::copy(odata, dev_data.ptr(), n);
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

		cu::cVec<int> dev_idata(n, idata), dev_bdata(n), dev_sdata(N), dev_odata(n);

		timer().startGpuTimer();
		
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		kern_bmap<<<fullBlocksPerGrid, blockSize>>>(n, dev_idata.raw_ptr(), dev_bdata.raw_ptr());

		cu::copy(dev_sdata.ptr(), dev_bdata.ptr(), n);
		scan_dev(N, &dev_sdata);

		kern_scatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_idata.raw_ptr(), dev_bdata.raw_ptr(),
			dev_sdata.raw_ptr(), dev_odata.raw_ptr());

		timer().endGpuTimer();

		cu::copy(odata, dev_odata.ptr(), n);

		for (int i = 0; i < n; i++)
			if (!odata[i])
				return i;
		
		return n;
	}
}
}
