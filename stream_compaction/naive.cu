#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "cVec.h"

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

	template <typename T>
	__global__ void kern_scan(int d, int n, const T *__restrict__ in, T *__restrict__ out)
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

		cu::cVec<int> data_in(n, idata), data_out(n);
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		timer().startGpuTimer();

		int log2n = ilog2ceil(n);
		for (int d = 0; d < log2n; d++) {
			kern_scan<<<fullBlocksPerGrid, blockSize>>>(d, n, data_in.raw_ptr(), data_out.raw_ptr());
			checkCUDAError("kern_scan failed!");
			std::swap(data_in, data_out);
		}

		timer().endGpuTimer();


		cu::copy(odata+1, data_in.ptr(), n-1);
		odata[0] = 0;
	}
}
}
