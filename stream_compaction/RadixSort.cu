#include "RadixSort.h"
#include <cuda.h>
#include <cuda_runtime.h>
namespace StreamCompaction {
	namespace RadixSort {

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)  // We can use defines provided in this project

		int* dev_buf;
		int* bufBit;
		int* falseBuf;
		int* trueBuf;
		int* bufNotBits;
		int* bufAnswer;
#define blockSize 128


		void AllocateMemory(int n)
		{
			cudaMalloc((void**)&dev_buf, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buf failed!");
			cudaMalloc((void**)&bufBit, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufloader failed!");
			cudaMalloc((void**)&falseBuf, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufB failed!");
			cudaMalloc((void**)&trueBuf, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufS failed!");
			cudaMalloc((void**)&bufNotBits, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufAnswers failed!");
			cudaMalloc((void**)&bufAnswer, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufAnswers failed!");
			cudaDeviceSynchronize();
		}

		void FreeMemory() {
			cudaFree(dev_buf);
			cudaFree(bufBit);
			cudaFree(falseBuf);
			cudaFree(trueBuf);
			cudaFree(bufNotBits);
			cudaFree(bufAnswer);
		}


		__global__ void PopulateBits(int bitOrder, int* bufInputData, int* bufBit, int N)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > N - 1)
			{
				return;
			}
			int mask = 1 << bitOrder;
			int masked_num = bufInputData[index] & mask;
			int thebit = masked_num >> bitOrder;
			bufBit[index] = thebit;
		}

		__global__ void PopulateNotBits(int *bitNotBits, const int* bufBits, int N)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > N - 1)
			{
				return;
			}
			bitNotBits[index] = ~bufBits[index];
		}

		__global__ void CopyAnswerToInputBuf(int* BufAnswer, int* InputBuf, int N)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > N - 1)
			{
				return;
			}
			InputBuf[index] = BufAnswer[index];
		}


		__global__ void ComputeTArray(int* outputBuf, int *falseBuf, int totalFalses, int N)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > N - 1)
			{
				return;
			}
			outputBuf[index] = index - falseBuf[index] + totalFalses;
		}

		__global__ void PerformScatter(int* outputBuf, int* inputBuf, int* bitBuf, int*falseBuf, int *trueBuf, int N)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > N - 1)
			{
				return;
			}
			if (bitBuf[index])
			{
				outputBuf[index] = trueBuf[index];
				return;
			}
			outputBuf[index] = falseBuf[index];

		}


		/*void PerformNormalSort(int n, int* odata, const int* idata)
		{

		}*/



		void PerformGPUSort(int n, int* odata, const int* idata)
		{
			AllocateMemory(n);
			cudaMemcpy(dev_buf, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			for (int i = 0; i < 6; i++)
			{
				PopulateBits << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, bufBit, n);
				cudaDeviceSynchronize();
				PopulateNotBits << < fullBlocksPerGrid, blockSize >> > (bufNotBits, bufBit, n);
				cudaDeviceSynchronize();

				int* inputNotBits= new int[n];
				cudaMemcpy(inputNotBits, bufNotBits, n * sizeof(int), cudaMemcpyDeviceToHost);
				Efficient::scan(n, odata, inputNotBits);
				cudaMemcpy(falseBuf, odata, n * sizeof(int), cudaMemcpyHostToDevice);

				int TotalFalses = inputNotBits[n - 1] + odata[n - 1];
				ComputeTArray << < fullBlocksPerGrid, blockSize >> > (trueBuf, falseBuf, TotalFalses, n);
				cudaDeviceSynchronize();
				PerformScatter << < fullBlocksPerGrid, blockSize >> > (bufAnswer, dev_buf, bufBit, falseBuf, trueBuf, n);
				cudaDeviceSynchronize();
				CopyAnswerToInputBuf << < fullBlocksPerGrid, blockSize >> > (bufAnswer, dev_buf, n);

			}

			cudaMemcpy(odata, dev_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);
			for (int i = 0; i < n; i++)
			{
				odata[i];
			}
		}

	}
}