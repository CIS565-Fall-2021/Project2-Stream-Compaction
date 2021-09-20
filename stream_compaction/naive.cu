#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
namespace StreamCompaction {
	namespace Naive {

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)  // We can use defines provided in this project


		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__
		int* dev_buf1;
		int* dev_buf2;
#define blockSize 1024

		__global__ void performScan(int d, int* buf1, int* buf2, int N)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			//int pow2_d = pow(2, d);
			int pow2_dminus1 = pow(2, d - 1);
			if (index >= pow2_dminus1)
			{
				buf2[index] = buf1[index - pow2_dminus1] + buf1[index];
			}
			else
			{
				buf2[index] = buf1[index];
			}

		}

		__global__ void ShiftRight(int* buf1, int* buf2, int N, int difference)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			if (index == 0)
			{
				buf2[index] = 0;
				return;
			}
			buf2[index] = buf1[index + difference - 1];

		}

		void FreeMemory() {
			cudaFree(dev_buf1);
			cudaFree(dev_buf2);
		}

		void AllocateMemory(int n)
		{
			cudaMalloc((void**)&dev_buf1, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");
			cudaMalloc((void**)&dev_buf2, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");
			cudaDeviceSynchronize();
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			timer().startGpuTimer();
			// TODO

			int power2 = 1;
			int nearesttwo = 2;

			for (int i = 0; i < n; i++)
			{
				nearesttwo = nearesttwo << 1;
				if (nearesttwo >= n)
				{
					break;
				}
			}
			int difference = nearesttwo - n;

			int finalMemSize = n + difference;

			int* arr_z = new int[finalMemSize];

			for (int i = 0; i < finalMemSize; i++)
			{
				if (i < difference)
				{
					arr_z[difference] = 0;
					continue;
				}
				arr_z[i] = idata[i - difference];
			}

			for (int i = 0; i < difference; i++)
			{
				arr_z[i] = 0;
			}
			for (int i = 0; i < n; i++)
			{
				arr_z[i + difference] = idata[i];
			}

	/*		printf(" \n Array Before:");
			for (int i = 0; i < finalMemSize; i++)
			{
				printf("%3d ", arr_z[i]);
			}
			printf("\n");*/
			int d = ilog2(finalMemSize);
			AllocateMemory(finalMemSize);
			cudaMemcpy(dev_buf1, arr_z, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);
			cudaDeviceSynchronize();
			for (int i = 1; i <= d; i++)
			{
				performScan << < fullBlocksPerGrid, blockSize >> > (i, dev_buf1, dev_buf2, finalMemSize);
				cudaDeviceSynchronize();
				std::swap(dev_buf1, dev_buf2);
			}

			ShiftRight << < fullBlocksPerGrid, blockSize >> > (dev_buf1, dev_buf2, finalMemSize, difference);
			cudaMemcpy(odata, dev_buf2, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

		
			/*printf(" \n Array After:");*/
			/*for (int i = 0; i < finalMemSize; i++)
			{
				printf("%3d ", arr_z[i]);
			}*/

			/*	 printf("]\n");
				 for (int i = 0; i < n; i++)
				 {
					 printf("%3d ", odata[i]);
				 }*/


			timer().endGpuTimer();
			cudaDeviceSynchronize();
			FreeMemory();
		}
	}
}
