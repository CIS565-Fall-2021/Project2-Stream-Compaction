#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)  // We can use defines provided in this project

        int* dev_buf;
        int* dev_bufB;
        int* dev_bufS;
        int* dev_buftemp;
        int* dev_bufAnswers;

#define blockSize 128

        __global__ void performUpSweep(int d, int* buf, int N)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int pow2_d = pow(2, d);
			int pow2_dplus1 = pow2_d * 2;
            if (index + pow2_dplus1 - 1 > N-1)
            {
                return;
            }

			if ((index + pow2_dplus1 - 1) % pow2_dplus1 == (pow2_dplus1-1))
			{
				buf[index + pow2_dplus1 - 1] += buf[index + pow2_d - 1];
			}

			if (index + pow2_dplus1 - 1 == N - 1)
			{
				buf[index + pow2_dplus1 - 1] = 0;
				return;
			}

        }

		__global__ void performDownSweep(int d, int* buf, int N)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int pow2_d = pow(2, d);
			int pow2_dplus1 = pow2_d * 2;
			if (index + pow2_dplus1 - 1> N -1)
			{
				return;
			}

			if ((index + pow2_dplus1 - 1) % pow2_dplus1 == (pow2_dplus1 - 1))
			{

				int t = buf[index + pow2_d - 1];
				buf[index + pow2_d - 1] = buf[index + pow2_dplus1 - 1];
				buf[index + pow2_dplus1 - 1] += t;
			}
		}

        void FreeMemoryScan() {
            cudaFree(dev_buf);
        }

        void AllocateMemory(int n)
        {
            cudaMalloc((void**)&dev_buf, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_buf failed!");
            cudaDeviceSynchronize();
        }

		void AllocateMemoryCompaction(int n)
		{
			cudaMalloc((void**)&dev_buf, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buf failed!");
			cudaMalloc((void**)&dev_bufB, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufB failed!");
			cudaMalloc((void**)&dev_bufS, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufS failed!");
			cudaMalloc((void**)&dev_bufAnswers, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufAnswers failed!");
			cudaDeviceSynchronize();
		}

		void FreeMemoryCompaction() {
			cudaFree(dev_buf);
			cudaFree(dev_bufB);
			cudaFree(dev_bufS);
			cudaFree(dev_bufAnswers);
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

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

			int bp = idata[0];
			int fp = idata[n - 1];
			for (int i = 0; i < difference; i++)
			{
				arr_z[i] = 0;
			}
			for (int i = 0; i < n; i++)
			{
				arr_z[i + difference] = idata[i];
			}

			/*   for (int i = 0; i < finalMemSize; i++)
			   {
				   printf("%3d ", arr_z[i]);
			   }
			   printf("\n");*/
			int d = ilog2ceil(finalMemSize);
			AllocateMemory(finalMemSize);
			cudaMemcpy(dev_buf, arr_z, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);

			for (int i = 0; i <= d -1; i++)
			{
				performUpSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, finalMemSize);
			}

			//dev_buf[finalMemSize-1] = 0;
		/*	cudaMemcpy(arr_z, dev_buf, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

			printf("\n Upsweep");
			for (int i = 0; i < finalMemSize; i++)
			{
				printf("%3d ", arr_z[i]);
			}
			printf("\n");*/

			for (int i = d-1; i >=0 ; i--)
			{
				performDownSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, finalMemSize);
			}


			cudaMemcpy(arr_z, dev_buf, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);
			//printf("\n Downsweep");
			//   for (int i = 0; i < finalMemSize; i++)
			//   {
			//	   printf("%3d ", arr_z[i]);
			//   }
			//   printf("\n");
			for (int i = 0; i < n; i++)
			{
				odata[i] = arr_z[i + difference];
			}

			 //printf("]\n");
			 //for (int i = 0; i < n; i++)
			 //{
				// printf("%3d ", odata[i]);
			 //}
            timer().endGpuTimer();

			FreeMemoryScan();
        }

		void scanWithoutTimer(int n, int* odata, const int* idata) {
			// TODO

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

			int bp = idata[0];
			int fp = idata[n - 1];
			for (int i = 0; i < difference; i++)
			{
				arr_z[i] = 0;
			}
			for (int i = 0; i < n; i++)
			{
				arr_z[i + difference] = idata[i];
			}

			/*   for (int i = 0; i < finalMemSize; i++)
			   {
				   printf("%3d ", arr_z[i]);
			   }
			   printf("\n");*/
			int d = ilog2ceil(finalMemSize);
			AllocateMemory(finalMemSize);
			cudaMemcpy(dev_buf, arr_z, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);

			for (int i = 0; i <= d - 1; i++)
			{
				performUpSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, finalMemSize);
			}

			//dev_buf[finalMemSize-1] = 0;
		/*	cudaMemcpy(arr_z, dev_buf, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

			printf("\n Upsweep");
			for (int i = 0; i < finalMemSize; i++)
			{
				printf("%3d ", arr_z[i]);
			}
			printf("\n");*/

			for (int i = d - 1; i >= 0; i--)
			{
				performDownSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, finalMemSize);
			}


			cudaMemcpy(arr_z, dev_buf, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);
			//printf("\n Downsweep");
			//   for (int i = 0; i < finalMemSize; i++)
			//   {
			//	   printf("%3d ", arr_z[i]);
			//   }
			//   printf("\n");
			for (int i = 0; i < n; i++)
			{
				odata[i] = arr_z[i + difference];
			}

			//printf("]\n");
			//for (int i = 0; i < n; i++)
			//{
			   // printf("%3d ", odata[i]);
			//}

			FreeMemoryScan();
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
			int finalMemSize = n;
			AllocateMemoryCompaction(finalMemSize);
			
			
			timer().startGpuTimer();
            // TODO
			cudaMemcpy(dev_buf, idata, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);


			Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (finalMemSize, dev_bufB, dev_buf);

			//Copy bool value to new array to perform scan
			int *arr_boolean = new int[finalMemSize];
			cudaMemcpy(arr_boolean, dev_bufB, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

		/*	printf("\n Bollean Result: \n");
			   for (int i = 0; i < finalMemSize; i++)
			   {
				   printf("%3d ", arr_boolean[i]);
			   }
			   printf("\n");*/

			//Create new array to store answers from scan
			int* arr_scanResult = new int[finalMemSize];
			scanWithoutTimer(finalMemSize, arr_scanResult, arr_boolean);

		/*	printf("\n Scan Result:\n");
			for (int i = 0; i < finalMemSize; i++)
			{
				printf("%3d ", arr_scanResult[i]);
			}
			printf("\n");*/

			//Copy the scan answers to Dev_BufS to further process
			cudaMemcpy(dev_bufS, arr_scanResult, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			int numElements = arr_scanResult[finalMemSize - 1];
			///Remove this dont know why but dev_buf values are being replaced by scan values
			cudaMalloc((void**)&dev_buftemp, finalMemSize * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buftemp failed!");
			cudaMemcpy(dev_buftemp, idata, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (finalMemSize, dev_bufAnswers, dev_buftemp,
				dev_bufB, dev_bufS);

			cudaMemcpy(odata, dev_bufAnswers, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

		/*	printf("\n Finals Result:");
			for (int i = 0; i < numElements; i++)
			{
				printf("%3d ", odata[i]);
			}
			printf("\n");*/
            timer().endGpuTimer();
			FreeMemoryCompaction();

			if (arr_boolean[finalMemSize - 1] == 1)
			{
				return numElements + 1; // Since indexing start from 0
			}

            return numElements; //if last element boolean is 0 its scan result include 1 extra sum counting for 0 index
        }
    }
}
