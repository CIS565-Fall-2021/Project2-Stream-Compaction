#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 256

namespace StreamCompaction {
namespace Naive {


	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	__global__ void kern_scan(int d, int n, const int *__restrict__ in, int *__restrict__ out)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;

		if (idx >= (1 << d))
			out[idx] = in[idx] + in[idx - (1 << d)];
		else
			out[idx] = in[idx];
	}

	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int* odata, const int* idata)
	{
		/* the implementation does inclusive scan, then the first (n-1) vals are copied
		 * to odata[1:] and odata[0] is set to 0
		*/
		int log2n = ilog2ceil(n);

		int *dev_data_in, *dev_data_out;

		cudaMalloc((void**)&dev_data_in, n * sizeof(*dev_data_in));
		checkCUDAError("cudaMalloc dev_data_in failed!");

		cudaMalloc((void**)&dev_data_out, n * sizeof(*dev_data_out));
		checkCUDAError("cudaMalloc dev_data_out failed!");

		cudaMemcpy(dev_data_in, idata, sizeof(*idata) * n, cudaMemcpyHostToDevice);
		checkCUDAError("memcpy idata to device failed!");


		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		timer().startGpuTimer();

		for (int d = 0; d < log2n; d++) {
			kern_scan<<<fullBlocksPerGrid, blockSize>>>(d, n, dev_data_in, dev_data_out);
			std::swap(dev_data_in, dev_data_out);
		}
		timer().endGpuTimer();

		checkCUDAError("kern_scan failed!");

		cudaMemcpy(odata+1, dev_data_in, (n-1) * sizeof(*odata), cudaMemcpyDeviceToHost);
		checkCUDAError("memcpy dev_data_in to host failed!");
		odata[0] = 0;

		cudaFree(dev_data_in);
		cudaFree(dev_data_out);
		checkCUDAError("cudaFree failed!");
	}
}
}
