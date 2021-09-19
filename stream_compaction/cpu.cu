#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // DONE-Part 1
            if (n < 0)
            {
                timer().endCpuTimer();
                return;
            }

            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // DONE-Part 1
            if (n < 0)
            {
                timer().endCpuTimer();
                return 0;
            }

            int count = 0;
            for (int i = 0; i < n; i++)
            {
                int data = idata[i];
                if (data != 0)
                {
                    odata[count] = data;
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // DONE-Part 1
            if (n < 0)
            {
                timer().endCpuTimer();
                return 0;
            }

            // Transform input into binary array
            int* valid_indices = new int[n];
            for (int i = 0; i < n; i++)
            {
                valid_indices[i] = (idata[i] == 0) ? 0 : 1;
            }

            // Run scan algorithm on valid index data
            // Please note that I could not call the scan function as it starts
            // the timer again and causes the program to crash.
            int *output_indices = new int[n];
            output_indices[0] = 0;
            for (int i = 1; i < n; i++)
            {
                output_indices[i] = output_indices[i - 1] + valid_indices[i - 1];
            }

            // Write valid data to output based on indices computed in scan
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (valid_indices[i] == 1)
                {
                    odata[output_indices[i]] = idata[i];
                    count++;
                }
            }

            delete[] valid_indices;
            delete[] output_indices;

            timer().endCpuTimer();
            return count;
        }
    }
}
