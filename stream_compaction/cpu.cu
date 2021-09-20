#include <cstdio>
#include "cpu.h"

#include "common.h"

#include <memory> // for smart pointers

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void scanWithoutTimer(int n, int* odata, const int* idata) {
            odata[0] = 0;
            for (int j = 1; j < n; j++)
            {
                odata[j] = odata[j - 1] + idata[j - 1];
            }
        }

        /**
         * CPU scan (exclusive prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanWithoutTimer(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         * This stream compaction method will remove 0s from an array of ints.
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // Given an array of elements, create a new array with all the 0s 
            // removed while preserving order
            int index = 0;
            for (int i = 0; i < n; i++)
            {
                int thisElement = idata[i];
                if (thisElement != 0)
                {
                    odata[index] = thisElement;
                    index++;
                }
            }
            timer().endCpuTimer();
            return n - index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         * This stream compaction method will remove 0s from an array of ints.
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int numElement = 0;
            std::unique_ptr<int[]>tempArray{ new int[n] };
            std::unique_ptr<int[]>scanResult{ new int[n] };
            for (int i = 0; i < n; i++)
            {
                scanResult[i] = -1;
            }

            // STEP 1: Compute temp Array with 0s and 1s
            // intialize array such that all elements meet criteria
            for (int i = 0; i < n; i++)
            {
                tempArray[i] = 1;
            }
            // next, figure out which one doesn't meet criteria
            for (int i = 0; i < n; i++)
            {
                // since we want to remove 0s, elements with value = 0 doesn't
                // meet criteria
                if (idata[i] == 0)
                {
                    tempArray[i] = 0;
                }
            }

            // STEP 2: Run exclusive scan on tempArray
            scanWithoutTimer(n, scanResult.get(), tempArray.get());

            // STEP 3: scatter
            for (int i = 0; i < n; i++)
            {
                // result of scan is index into final array
                int index = scanResult[i];
                // only write an element if temp array has a 1
                if (tempArray[i] == 1)
                {
                    odata[index] = idata[i];
                    numElement++;
                }
            }

            timer().endCpuTimer();
            return n - numElement;
        }
    }
}
