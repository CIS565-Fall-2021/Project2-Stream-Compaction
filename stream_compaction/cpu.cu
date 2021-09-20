#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction
{
    namespace CPU
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            // TODO
            int tmpAcc = 0;
            for (int i = 0; i < n; i++)
            {
                odata[i] = tmpAcc;
                tmpAcc += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            // TODO
            int curIdx = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[curIdx] = idata[i];
                    curIdx++;
                }
            }
            timer().endCpuTimer();
            return curIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata)
        {
            int *tmpData = new int[n];
            int *tmpData2 = new int[n];
            timer().startCpuTimer();
            // map
            for (int i = 0; i < n; i++)
            {
                tmpData[i] = idata[i] != 0;
            }
            // scan
            int tmpAcc = 0;
            for (int i = 0; i < n; i++)
            {
                tmpData2[i] = tmpAcc;
                tmpAcc += tmpData[i];
            }
            int const *arrPtr = idata;
            // if last elem of mapped boolarr is 0, tmpData[n-1] is 0
            int retVal = tmpData2[n - 1] + tmpData[n - 1];
            for (int i = 0; i < retVal; i++)
            {
                while (*arrPtr == 0)
                {
                    arrPtr++;
                }
                odata[i] = *arrPtr;
                arrPtr++;
            }
            timer().endCpuTimer();
            delete tmpData2;
            delete tmpData;
            return retVal;
        }
    }
}
