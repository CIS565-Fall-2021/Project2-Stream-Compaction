#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

//#define WRITE_EXC_WITH_INC 1

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweepOnce(int n, int* odata, int stride) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int base = index * stride;
            int targetOffset = stride - 1;
            int targetIdx = base + targetOffset;
            if (targetIdx >= n) {
                return;
            }
            int leftIdx = base + (targetOffset >> 1);

            odata[targetIdx] += odata[leftIdx];
        }

        __global__ void kernDownSweepOnce(int n, int* odata, int stride) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int base = index * stride;
            int targetOffset = stride - 1;
            int targetIdx = base + targetOffset;
            if (targetIdx >= n) {
                return;
            }
            int leftIdx = base + (targetOffset >> 1);
            
            int prevLeftValue = odata[leftIdx];
            odata[leftIdx] = odata[targetIdx];
            odata[targetIdx] += prevLeftValue;
        }

        void scanGpu(int newN, int logn, int* cuda_g_odata, int threadsPerBlock = 128) {
            // 1 Up sweep.
            int stride = 1;
            int blockCount = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            int iToAddBlockBelow = logn;
            for (int i = 0; i < logn; ++i) {
                stride <<= 1;
                if (blockCount == 1 && iToAddBlockBelow == logn) {
                    iToAddBlockBelow = i;
                }
                blockCount = std::max(1, blockCount >> 1);
                //printf("UP blockCount:%d, stride:%d\n", blockCount, stride);
                kernUpSweepOnce<<<blockCount, threadsPerBlock>>>(newN, cuda_g_odata, stride);
            }

            // 2 Down sweep.
            cudaMemset(cuda_g_odata + newN - 1, 0, sizeof(int));

            blockCount = 1;
            for (int i = logn - 1; i >= 0; --i) {
                //printf("DOWN blockCount:%d, stride:%d\n", blockCount, stride);
                kernDownSweepOnce<<<blockCount, threadsPerBlock>>>(newN, cuda_g_odata, stride);
                stride = std::max(1, stride >> 1);

                if (i < iToAddBlockBelow) {
                    blockCount <<= 1;
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int threadsPerBlock = 128;

            int* cuda_g_odata = nullptr;
            int logn = ilog2ceil(n);
            size_t newN = 1i64 << logn;
            size_t sizeNewN = sizeof(int) * newN;
            cudaMalloc(&cuda_g_odata, sizeNewN);
            cudaMemset(cuda_g_odata, 0, sizeNewN);
            cudaMemcpy(cuda_g_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            // DONE
            scanGpu(newN, logn, cuda_g_odata, threadsPerBlock);
            timer().endGpuTimer();

            cudaMemcpy(odata, cuda_g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_odata);
        }

        void compactGpu(int newN, int logn, int* cuda_g_odata, int* cuda_g_bools, int* cuda_g_indices, const int* cuda_g_idata, int threadsPerBlock = 128) {
            int blockCountSimplePara = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            Common::kernMapToBoolean<<<blockCountSimplePara, threadsPerBlock>>>(newN, cuda_g_bools, cuda_g_idata);
            cudaMemcpy(cuda_g_indices, cuda_g_bools, sizeof(int) * newN, cudaMemcpyDeviceToDevice);
            scanGpu(newN, logn, cuda_g_indices, threadsPerBlock);
            Common::kernScatter<<<blockCountSimplePara, threadsPerBlock>>>(newN, cuda_g_odata, cuda_g_idata, cuda_g_bools, cuda_g_indices);
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
            int threadsPerBlock = 128;

            int* cuda_g_odata = nullptr;
            int* cuda_g_idata = nullptr;
            int* cuda_g_bools = nullptr;
            int* cuda_g_indices = nullptr;
            int logn = ilog2ceil(n);
            size_t newN = 1i64 << logn;
            size_t sizeNewN = sizeof(int) * newN;
            cudaMalloc(&cuda_g_odata, sizeNewN);
            cudaMalloc(&cuda_g_idata, sizeNewN);
            cudaMalloc(&cuda_g_bools, sizeNewN);
            cudaMalloc(&cuda_g_indices, sizeNewN);
            cudaMemset(cuda_g_odata, 0, sizeNewN);
            cudaMemcpy(cuda_g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // DONE
            compactGpu(newN, logn, cuda_g_odata, cuda_g_bools, cuda_g_indices, cuda_g_idata, threadsPerBlock);
            timer().endGpuTimer();

            int inclusivePrefixSum = 0, lastEle = 0;
            cudaMemcpy(&inclusivePrefixSum, cuda_g_indices + newN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastEle, cuda_g_bools + newN - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, cuda_g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_odata);
            cudaFree(cuda_g_idata);
            cudaFree(cuda_g_bools);
            cudaFree(cuda_g_indices);

            return inclusivePrefixSum + lastEle;
        }

        //////////////////////////////////////////////////////////////////

        //__global__ void kernInclusiveScanNaiveInBlock(int n, int* odata) {
        //    extern __shared__ int sharedMemory[];
        //    int* s_idata = sharedMemory;
        //    //int* s_idata2 = sharedMemory + threadIdx.x;
        //    int indexBaseInBlock = blockIdx.x * blockDim.x;
        //    int localIdx = threadIdx.x;

        //    int totalIdx = indexBaseInBlock + localIdx;
        //    if (totalIdx >= n) {
        //        return;
        //    }

        //    s_idata[localIdx] = odata[totalIdx];

        //    // Loop in kernel
        //    for(unsigned int stride = 1; stride <= n; stride <<= 1) {
        //        __syncthreads();
        //        //int* tmp = s_idata2;
        //        //s_idata2 = s_idata;
        //        //s_idata = tmp;
        //        //printf("stride:%d, blockCount:%d, threadsPerBlock:%d\n", stride, gridDim.x, blockDim.x);
        //        int fromLocalIdx = localIdx - stride;
        //        //int prevOdata = fromLocalIdx < 0 ? 0 : s_idata2[fromLocalIdx];
        //        int prevOdata = fromLocalIdx < 0 ? 0 : s_idata[fromLocalIdx];
        //        __syncthreads();

        //        //s_idata[localIdx] = s_idata2[localIdx] + prevOdata;
        //        s_idata[localIdx] += prevOdata;
        //        //__syncthreads();
        //    }
        //    odata[totalIdx] = s_idata[localIdx];


        //    // Loop out of kernel
        //    //int indexBaseInBlock = blockIdx.x * blockDim.x;
        //    //int localIdx = threadIdx.x;

        //    //int totalIdx = indexBaseInBlock + localIdx;
        //    //if (totalIdx >= n) {
        //    //    return;
        //    //}

        //    //s_idata[localIdx] = odata[totalIdx];
        //    //__syncthreads();
        //    //int fromLocalIdx = localIdx - stride;
        //    //int prevOdata = fromLocalIdx < 0 ? 0 : s_idata[fromLocalIdx];

        //    //odata[totalIdx] = s_idata[localIdx] + prevOdata;
        //}

        //__global__ void kernInclusiveScanSerialInBlock(int n, int *odata) {
        //    int bkDim = blockDim.x;
        //    int index = ((blockIdx.x * bkDim) + threadIdx.x) * bkDim;
        //    if (index >= n) {
        //        return;
        //    }
        //    int loopLimit = n > bkDim ? bkDim : n;
        //    int sum = 0;
        //    for (int i = 0; i < loopLimit; ++i) {
        //        int totalIndex = index + i;
        //        int temp = odata[totalIndex];
        //        sum += temp;
        //        odata[totalIndex] = sum;
        //    }
        //}

        __global__ void kernInclusiveScanWorkEfficientInBlock(int n, int* odata) {
            extern __shared__ int sharedMemory[];
            int bkDim = blockDim.x;
            int dblBkDim = blockDim.x << 1;
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            unsigned int loopLimit = (n > dblBkDim) ? dblBkDim : n;
            unsigned int halfLoopLimit = loopLimit >> 1;

            int indexBaseInBlock = bid * dblBkDim;
            int localIdx = ((tid + 1) << 1) - 1;

            int totalIdx = indexBaseInBlock + localIdx;
            if (totalIdx >= n) {
                return;
            }
            int* s_idata = sharedMemory;
            int* s_odata = sharedMemory + dblBkDim;

            int localIdxToRead = tid;
            int localIdxToRead2 = tid + halfLoopLimit;

            // Bank conflict?
            s_idata[localIdxToRead] = odata[indexBaseInBlock + localIdxToRead];
            s_idata[localIdxToRead2] = odata[indexBaseInBlock + localIdxToRead2];
            s_odata[localIdxToRead] = s_idata[localIdxToRead];
            s_odata[localIdxToRead2] = s_idata[localIdxToRead2];
            //__syncthreads();

            unsigned int stride = 2;

            // Up sweep. It is not necessary to calculate the rightmost sum
            //for (; stride <= loopLimit; stride <<= 1) {
            for (; stride < loopLimit; stride <<= 1) {
                localIdx = tid * stride + (stride - 1);
                __syncthreads();
                if (localIdx < loopLimit) {
                    int leftLocalIdx = tid * stride + ((stride - 1) >> 1);
                    s_odata[localIdx] += s_odata[leftLocalIdx];
                }
            }
            //printf("tid:%d, stride:%d\n", tid, stride);
            
            //stride >>= 1; // Because the rightmost sum is not calculated, we don't need to halve the stride
            if (tid * stride + stride == loopLimit) {
                s_odata[loopLimit - 1] = 0;
            }

            // Down sweep
            for (; stride > 1; stride >>= 1) {
                localIdx = tid * stride + (stride - 1);
                __syncthreads();
                if (localIdx < loopLimit) {
                    int leftLocalIdx = tid * stride + ((stride - 1) >> 1);
                    int tmp = s_odata[leftLocalIdx];
                    s_odata[leftLocalIdx] = s_odata[localIdx];
                    s_odata[localIdx] += tmp;
                }
            }

            // Exclusive to inclusive
            __syncthreads();
            s_odata[localIdxToRead] += s_idata[localIdxToRead];
            s_odata[localIdxToRead2] += s_idata[localIdxToRead2];

            // Write back
            odata[indexBaseInBlock + localIdxToRead] = s_odata[localIdxToRead];
            odata[indexBaseInBlock + localIdxToRead2] = s_odata[localIdxToRead2];
        }

        inline void inclusiveScanInBlock(int newN, int* cuda_g_odata, int threadsPerBlock = 128) {
            //for(int stride = 1; stride <= n; stride <<= 1) {
            //    int blockCount = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            //    //printf("stride:%d, blockCount:%d, threadsPerBlock:%d\n", stride, blockCount, threadsPerBlock);
            //    kernInclusiveScanInBlockOnce<<<blockCount, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(newN, cuda_g_odata, stride);
            //}
            //int blockCount = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            //printf("stride:%d, blockCount:%d, threadsPerBlock:%d\n", stride, blockCount, threadsPerBlock);
            //kernInclusiveScanInBlock<<<blockCount, threadsPerBlock, threadsPerBlock * sizeof(int) * 2>>>(newN, cuda_g_odata);
            //kernInclusiveScanNaiveInBlock<<<blockCount, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(newN, cuda_g_odata);

            int blockCount = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            kernInclusiveScanWorkEfficientInBlock<<<blockCount, threadsPerBlock >> 1, (threadsPerBlock << 1) * sizeof(int)>>>(newN, cuda_g_odata);
        }

        __global__ void kernGenBlockSums(int newN, int* blockSums, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= newN) {
                return;
            }
            int sumIdx = index * blockDim.x + blockDim.x - 1;
            blockSums[index] = idata[sumIdx];
        }

        //__global__ void kernGenBlockIncrementsToExclusive(int newN, int* blockIncrements, const int* idata, const int* blockSums) {
        //    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        //    if (index >= newN) {
        //        return;
        //    }
        //    int bkIdx = blockIdx.x;
        //    int notSetZero = (index + 1 < newN);
        //    blockIncrements[notSetZero * index + notSetZero] = notSetZero * (blockSums[bkIdx] + idata[index]);
        //}

        __global__ void kernGenBlockIncrementsToInclusive(int newN, int* blockIncrements, const int* idata, const int* blockSums) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= newN) {
                return;
            }
            int bkIdx = blockIdx.x;
            blockIncrements[index] = blockSums[bkIdx] + idata[index];
        }

#if WRITE_EXC_WITH_INC
        __global__ void kernGenBlockIncrementsToIncExc(int newN, int* blockIncrements, int* odata, const int* blockSums) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= newN) {
                return;
            }
            int bkIdx = blockIdx.x;

            int idataAtIdx = odata[index];

            int blkIncAtIdx = blockSums[bkIdx] + idataAtIdx;
            __syncthreads();//Bug, maybe when ptr not in one block?

            blockIncrements[index] = blkIncAtIdx;
            int nonZeroIdx = (index + 1 < newN);
            odata[nonZeroIdx * (index + 1)] = nonZeroIdx * blkIncAtIdx;
        }
#endif // WRITE_EXC_WITH_INC

        //__global__ void kernIncToExc(int newN, int* odata, const int* idata) {
        //    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        //    if (index >= newN) {
        //        return;
        //    }
        //    int bkIdx = blockIdx.x;
        //    int notSetZero = index + 1 < newN;
        //    odata[notSetZero * index + notSetZero] = notSetZero * idata[index];
        //}

        void scanGpuBatch(int newN, int logn, int* cuda_g_odata, int* cuda_g_blockSums, int* cuda_g_blockIncrements, int threadsPerBlock = 128,
                int depth = 0, bool splitOnce = false) {
            int batch = (newN + threadsPerBlock - 1) / threadsPerBlock;
            //printf("N:%d, blockCount:%d, threadsPerBlock:%d, logn:%d\n", newN, batch, threadsPerBlock, logn);//TEST
            if (batch > 0) {
                if (batch > 1) {
                    //if (depth < 2) {
                    //    int serialBatch = std::max(1, batch / threadsPerBlock);
                    //    kernInclusiveScanSerialInBlock<<<serialBatch, threadsPerBlock>>>(newN, cuda_g_odata);
                    //}
                    //else {
                        inclusiveScanInBlock(newN, cuda_g_odata, threadsPerBlock);
                    //}

                    int nextNewN = batch;
                    int nextBatch = (nextNewN + threadsPerBlock - 1) / threadsPerBlock;
                    int logThPerBk = ilog2ceil(threadsPerBlock);
                    if (splitOnce) {
                        kernGenBlockSums<<<nextBatch, threadsPerBlock>>>(nextNewN, cuda_g_blockSums, cuda_g_odata);
                        scanGpu(nextNewN, logn - logThPerBk, cuda_g_blockSums, threadsPerBlock);
                        //kernGenBlockIncrementsToExclusive<<<batch, threadsPerBlock>>>(newN, cuda_g_blockIncrements, cuda_g_odata, cuda_g_blockSums);
                        //cudaMemcpy(cuda_g_odata, cuda_g_blockIncrements, sizeof(int) * (newN), cudaMemcpyDeviceToDevice);
#if WRITE_EXC_WITH_INC
                        kernGenBlockIncrementsToIncExc<<<batch, threadsPerBlock>>>(newN, cuda_g_blockIncrements, cuda_g_odata, cuda_g_blockSums);
#else // WRITE_EXC_WITH_INC
                        kernGenBlockIncrementsToInclusive<<<batch, threadsPerBlock>>>(newN, cuda_g_blockIncrements, cuda_g_odata, cuda_g_blockSums);
                        cudaMemset(cuda_g_odata, 0, sizeof(int));
                        cudaMemcpy(cuda_g_odata + 1, cuda_g_blockIncrements, sizeof(int) * (newN - 1), cudaMemcpyDeviceToDevice);
#endif // WRITE_EXC_WITH_INC
                    }
                    else {
                        kernGenBlockSums<<<nextBatch, threadsPerBlock>>>(nextNewN, cuda_g_blockSums, cuda_g_odata);
                        scanGpuBatch(nextNewN, logn - logThPerBk, cuda_g_blockSums, cuda_g_blockSums + nextNewN, cuda_g_blockIncrements, threadsPerBlock,
                            depth + 1, splitOnce);
                        //kernGenBlockIncrementsToExclusive<<<batch, threadsPerBlock>>>(newN, cuda_g_blockIncrements, cuda_g_odata, cuda_g_blockSums);
                        //cudaMemcpy(cuda_g_odata, cuda_g_blockIncrements, sizeof(int) * (newN), cudaMemcpyDeviceToDevice);
#if WRITE_EXC_WITH_INC
                        kernGenBlockIncrementsToIncExc<<<batch, threadsPerBlock>>>(newN, cuda_g_blockIncrements, cuda_g_odata, cuda_g_blockSums);
#else // WRITE_EXC_WITH_INC
                        kernGenBlockIncrementsToInclusive<<<batch, threadsPerBlock>>>(newN, cuda_g_blockIncrements, cuda_g_odata, cuda_g_blockSums);
                        cudaMemset(cuda_g_odata, 0, sizeof(int));
                        cudaMemcpy(cuda_g_odata + 1, cuda_g_blockIncrements, sizeof(int) * (newN - 1), cudaMemcpyDeviceToDevice);
#endif // WRITE_EXC_WITH_INC
                    }
                }
                else {
                    //inclusiveScanInBlock(newN, cuda_g_blockIncrements, threadsPerBlock); // Not correct if in depth > 0
                    inclusiveScanInBlock(newN, cuda_g_odata, threadsPerBlock);

                    cudaMemcpy(cuda_g_blockIncrements, cuda_g_odata, sizeof(int) * (newN), cudaMemcpyDeviceToDevice);
                    cudaMemset(cuda_g_odata, 0, sizeof(int));
                    cudaMemcpy(cuda_g_odata + 1, cuda_g_blockIncrements, sizeof(int) * (newN - 1), cudaMemcpyDeviceToDevice);
                }
            }
        }

        /**
        * Performs prefix-sum (aka scan) on idata, storing the result into odata.
        */
        void scanBatch(int n, int *odata, const int *idata, bool splitOnce) {
            int threadsPerBlock = 512;//128;

            int* cuda_g_odata = nullptr;
            int* cuda_g_blockSums = nullptr;
            int* cuda_g_blockIncrements = nullptr;

            int logn = ilog2ceil(n);
            size_t newN = 1i64 << logn;
            size_t sizeNewN = sizeof(int) * newN;
            cudaMalloc(&cuda_g_odata, sizeNewN);
            cudaMalloc(&cuda_g_blockSums, sizeNewN);
            cudaMalloc(&cuda_g_blockIncrements, sizeNewN);
            //cudaMemset(cuda_g_odata, 0, sizeNewN);
            cudaMemset(cuda_g_blockSums, 0, sizeNewN);
            //cudaMemset(cuda_g_blockIncrements, 0, sizeNewN);
            cudaMemcpy(cuda_g_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_g_blockIncrements, cuda_g_odata, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            timer().startGpuTimer();
            // DONE
            scanGpuBatch(newN, logn, cuda_g_odata, cuda_g_blockSums, cuda_g_blockIncrements, threadsPerBlock, 0, splitOnce);
            timer().endGpuTimer();

            cudaMemcpy(odata, cuda_g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_blockIncrements);
            cudaFree(cuda_g_blockSums);
            cudaFree(cuda_g_odata);
        }

        void compactGpuBatch(int newN, int logn, int* cuda_g_odata, int* cuda_g_bools, int* cuda_g_indices, int* cuda_g_blockSums, int* cuda_g_blockIncrements, 
                const int* cuda_g_idata, int threadsPerBlock = 128, bool splitOnce = false) {
            int blockCountSimplePara = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            Common::kernMapToBoolean<<<blockCountSimplePara, threadsPerBlock>>>(newN, cuda_g_bools, cuda_g_idata);
            cudaMemcpy(cuda_g_indices, cuda_g_bools, sizeof(int) * newN, cudaMemcpyDeviceToDevice);
            scanGpuBatch(newN, logn, cuda_g_indices, cuda_g_blockSums, cuda_g_blockIncrements, threadsPerBlock, 0, splitOnce);
            Common::kernScatter<<<blockCountSimplePara, threadsPerBlock>>>(newN, cuda_g_odata, cuda_g_idata, cuda_g_bools, cuda_g_indices);
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
        int compactBatch(int n, int *odata, const int *idata, bool splitOnce) {
            int threadsPerBlock = 512;//128

            int* cuda_g_odata = nullptr;
            int* cuda_g_idata = nullptr;
            int* cuda_g_bools = nullptr;
            int* cuda_g_indices = nullptr;

            int* cuda_g_blockSums = nullptr;
            int* cuda_g_blockIncrements = nullptr;

            int logn = ilog2ceil(n);
            size_t newN = 1i64 << logn;
            size_t sizeNewN = sizeof(int) * newN;
            cudaMalloc(&cuda_g_odata, sizeNewN);
            cudaMalloc(&cuda_g_idata, sizeNewN);
            cudaMalloc(&cuda_g_bools, sizeNewN);
            cudaMalloc(&cuda_g_indices, sizeNewN);
            cudaMemset(cuda_g_odata, 0, sizeNewN);
            cudaMemcpy(cuda_g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            cudaMalloc(&cuda_g_blockSums, sizeNewN);
            cudaMalloc(&cuda_g_blockIncrements, sizeNewN);

            cudaMemset(cuda_g_blockSums, 0, sizeNewN);
            cudaMemset(cuda_g_blockIncrements, 0, sizeNewN);

            timer().startGpuTimer();
            // DONE
            compactGpuBatch(newN, logn, cuda_g_odata, cuda_g_bools, cuda_g_indices, cuda_g_blockSums, cuda_g_blockIncrements, cuda_g_idata, threadsPerBlock);
            timer().endGpuTimer();

            int inclusivePrefixSum = 0, lastEle = 0;
            cudaMemcpy(&inclusivePrefixSum, cuda_g_indices + newN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastEle, cuda_g_bools + newN - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, cuda_g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_blockIncrements);
            cudaFree(cuda_g_blockSums);

            cudaFree(cuda_g_odata);
            cudaFree(cuda_g_idata);
            cudaFree(cuda_g_bools);
            cudaFree(cuda_g_indices);

            return inclusivePrefixSum + lastEle;
        }

        ///////// Sort

        __global__ void kernMapToBooleanWithBit(int n, int* boolOnes, int* boolZeros, const int *idata, int bit) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int nonZero = (idata[index] & (1 << bit)) >> bit;
                boolOnes[index] = nonZero;
                boolZeros[index] = 1 - nonZero;
            }
        }

        __global__ void kernScatter01(int n, int *odata,
                const int *idata, const int* boolOnes, const int* boolZeros, const int* indexZeros) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int scatterIdx = indexZeros[index];
                if (boolZeros[index] == 1) {
                    odata[scatterIdx] = idata[index];
                }
                if (boolOnes[index] == 1) {
                    int lastIndexZero = indexZeros[n - 1];
                    int lastBoolZero = boolZeros[n - 1];
                    int totalZeros = lastIndexZero + lastBoolZero;
                    odata[index - scatterIdx + totalZeros] = idata[index];
                }
            }
        }

        void sortGpu(int newN, int logn, int* cuda_g_odata, int* cuda_g_boolOnes, int* cuda_g_boolZeros, int* cuda_g_indexOnes, int* cuda_g_indexZeros, int* cuda_g_blockSums, int* cuda_g_blockIncrements, 
                int* cuda_g_idata, int threadsPerBlock = 128, bool splitOnce = false) {
            int blockCountSimplePara = (newN + (threadsPerBlock - 1)) / threadsPerBlock;
            // Suppose all nums are non-negative, so bit = 31 is not necessary.
            for (int bit = 0; bit < 31; ++bit) {
                if (bit > 0) {
                    std::swap(cuda_g_odata, cuda_g_idata);
                }
                kernMapToBooleanWithBit<<<blockCountSimplePara, threadsPerBlock>>>(newN, cuda_g_boolOnes, cuda_g_boolZeros, cuda_g_idata, bit);
                cudaMemcpy(cuda_g_indexZeros, cuda_g_boolZeros, sizeof(int) * newN, cudaMemcpyDeviceToDevice);
                cudaMemcpy(cuda_g_indexOnes, cuda_g_boolOnes, sizeof(int) * newN, cudaMemcpyDeviceToDevice);
                scanGpuBatch(newN, logn, cuda_g_indexZeros, cuda_g_blockSums, cuda_g_blockIncrements, threadsPerBlock, 0, splitOnce);
                //scanGpuBatch(newN, logn, cuda_g_indexOnes, cuda_g_blockSums, cuda_g_blockIncrements, threadsPerBlock, 0, splitOnce);
                kernScatter01<<<blockCountSimplePara, threadsPerBlock>>>(newN, cuda_g_odata, cuda_g_idata, cuda_g_boolOnes, cuda_g_boolZeros, cuda_g_indexZeros);
            }
        }

        void sort(int n, int* odata, const int* idata) {
            int threadsPerBlock = 512;//128

            int* cuda_g_odata = nullptr;
            int* cuda_g_idata = nullptr;
            int* cuda_g_boolOnes = nullptr;
            int* cuda_g_indexOnes = nullptr;
            int* cuda_g_boolZeros = nullptr;
            int* cuda_g_indexZeros = nullptr;

            int* cuda_g_blockSums = nullptr;
            int* cuda_g_blockIncrements = nullptr;

            int logn = ilog2ceil(n);
            size_t newN = 1i64 << logn;
            size_t sizeNewN = sizeof(int) * newN;
            cudaMalloc(&cuda_g_odata, sizeNewN);
            cudaMalloc(&cuda_g_idata, sizeNewN);
            cudaMalloc(&cuda_g_boolOnes, sizeNewN);
            cudaMalloc(&cuda_g_indexOnes, sizeNewN);
            cudaMalloc(&cuda_g_boolZeros, sizeNewN);
            cudaMalloc(&cuda_g_indexZeros, sizeNewN);
            cudaMemset(cuda_g_odata, 0x7FFFFFFF, sizeNewN);
            cudaMemcpy(cuda_g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(cuda_g_idata + n, 0x7FFFFFFF, sizeof(int) * (newN - n));

            cudaMalloc(&cuda_g_blockSums, sizeNewN);
            cudaMalloc(&cuda_g_blockIncrements, sizeNewN);

            cudaMemset(cuda_g_blockSums, 0, sizeNewN);
            cudaMemset(cuda_g_blockIncrements, 0x7FFFFFFF, sizeNewN);

            timer().startGpuTimer();
            // DONE
            sortGpu(newN, logn, cuda_g_odata, cuda_g_boolOnes, cuda_g_boolZeros, cuda_g_indexOnes, cuda_g_indexZeros, cuda_g_blockSums, cuda_g_blockIncrements, cuda_g_idata, threadsPerBlock, false);
            timer().endGpuTimer();

            //int inclusivePrefixSum = 0, lastEle = 0;
            //cudaMemcpy(&inclusivePrefixSum, cuda_g_indices + newN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(&lastEle, cuda_g_bools + newN - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, cuda_g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_blockIncrements);
            cudaFree(cuda_g_blockSums);

            cudaFree(cuda_g_odata);
            cudaFree(cuda_g_idata);
            cudaFree(cuda_g_boolZeros);
            cudaFree(cuda_g_indexZeros);
            cudaFree(cuda_g_boolOnes);
            cudaFree(cuda_g_indexOnes);
        }
    }

    namespace EfficientTest {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void scan(int n, int* odata, const int* idata) {
            int threadsPerBlock = 128;

            int logn = ilog2ceil(n);
            int newN = 1 << logn;

            int* cuda_g_odata = nullptr;
            //int* cuda_g_idata = nullptr;
            cudaMalloc(&cuda_g_odata, sizeof(int) * newN);
            //cudaMalloc(&cuda_g_idata, sizeof(int) * newN);

            //cudaMemset(cuda_g_odata, 0, sizeof(odata) * newN);
            cudaMemcpy(cuda_g_odata, idata, sizeof(int) * (newN), cudaMemcpyHostToDevice);

            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // DONE
            Efficient::inclusiveScanInBlock(newN, cuda_g_odata, threadsPerBlock);
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, cuda_g_odata, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(cuda_g_odata);
        }
    }
}
