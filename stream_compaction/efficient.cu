#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

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
	void scan_dev(int N, int *dev_data) {
		int log2n = ilog2ceil(N);
		printf("N: %d\tlog2n: %d\n", N, log2n);
		for (int d = 0; d < log2n; d++) {
			int count = N / (1 << d);
			dim3 fullBlocksPerGrid((count + blockSize - 1) / blockSize);
			kern_up_sweep<<<fullBlocksPerGrid, blockSize>>>(d, N, dev_data);
		}
		cudaMemset(dev_data+(N-1), 0, sizeof(*dev_data));
		for (int d = log2n - 1; d >= 0; d--) {
			int count = N / (1 << d);
			dim3 fullBlocksPerGrid((count + blockSize - 1) / blockSize);
			kern_down_sweep<<<fullBlocksPerGrid, blockSize>>>(d, N, dev_data);
		}
	}


	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata) {
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		int *dev_data;

		cudaMalloc((void**)&dev_data, N * sizeof(*dev_data));
		checkCUDAError("cudaMalloc dev_data failed!");

		cudaMemcpy(dev_data, idata, sizeof(*idata) * n, cudaMemcpyHostToDevice);
		checkCUDAError("memcpy idata to device failed!");
	

//		timer().startGpuTimer();

		scan_dev(N, dev_data);

//		timer().endGpuTimer();

		cudaMemcpy(odata, dev_data, sizeof(*odata) * n, cudaMemcpyDeviceToHost);
		checkCUDAError("memcpy device to odata failed!1");

		cudaFree(dev_data);
		checkCUDAError("cudaFree failed!");
	}


	__global__ void kern_bmap(int n, int *__restrict__ idata, int *__restrict__ bdata)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		
		if (idx >= n)
			return;

		bdata[idx] = bool(idata[idx]);
	}




	__global__ void kern_scatter(int n, int *__restrict__ idata, int *__restrict__ bdata,
					int *__restrict__ sdata, int *__restrict odata)
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
		int *dev_idata, *dev_bdata, *dev_sdata, *dev_odata;

		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		cudaMalloc((void**)&dev_idata, n * sizeof(*dev_idata));
		checkCUDAError("cudaMalloc dev_idata failed!");

		cudaMalloc((void**)&dev_bdata, n * sizeof(*dev_bdata));
		checkCUDAError("cudaMalloc dev_bdata failed!");

		cudaMalloc((void**)&dev_sdata, N * sizeof(*dev_sdata));
		checkCUDAError("cudaMalloc dev_sdata failed!");

		cudaMalloc((void**)&dev_odata, n * sizeof(*dev_odata));
		checkCUDAError("cudaMalloc dev_odata failed!");


		cudaMemcpy(dev_idata, idata, sizeof(*idata) * n, cudaMemcpyHostToDevice);
		checkCUDAError("memcpy idata to device failed!");


		timer().startGpuTimer();
		
		/* map to bools */
		printf("start\n");
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		kern_bmap<<<fullBlocksPerGrid, blockSize>>>(n, dev_idata, dev_bdata);
		printf("bool\n");
		cudaMemcpy(dev_sdata, dev_bdata, n * sizeof(*dev_bdata), cudaMemcpyDeviceToDevice);
		scan_dev(N, dev_sdata);
		printf("compact_scan\n");
		kern_scatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_idata, dev_bdata, dev_sdata, dev_odata);

		timer().endGpuTimer();


		cudaMemcpy(odata, dev_odata, n * sizeof(*odata), cudaMemcpyDeviceToHost);
		checkCUDAError("memcpy device to odata failed!");


		cudaFree(dev_idata);
		checkCUDAError("cudaFree failed!");
		cudaFree(dev_bdata);
		checkCUDAError("cudaFree failed!");
		cudaFree(dev_sdata);
		checkCUDAError("cudaFree failed!");
		cudaFree(dev_odata);
		checkCUDAError("cudaFree failed!");

		for (int i = 0; i < n; i++)
			if (!odata[i])
				return i;
		
		return n;
	}
}
}
