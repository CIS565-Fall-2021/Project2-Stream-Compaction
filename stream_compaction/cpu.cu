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
        void scan(int n, int *odata, const int *idata, bool timing_on) {
            if (timing_on) {
                timer().startCpuTimer();
            }

            // Exclusive scan
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            if (timing_on) {
                timer().endCpuTimer();
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // Simply pick non-zero elements
            int num_element = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[num_element] = idata[i];
                    num_element++;
                }             
            }

            timer().endCpuTimer();
            return num_element;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            // Memory allocation
            int *bool_buffer = new int[n];
            int *scan_buffer = new int[n];

            timer().startCpuTimer();
            
            // Set 1 for non-zero elements
            for (int i = 0; i < n; i++) {
                bool_buffer[i] = (idata[i] != 0);
            }

            // Scan
            scan(n, scan_buffer, bool_buffer, false);

            // Scatter
            for (int i = 0; i < n; i++) {
                if (bool_buffer[i]) {
                    odata[scan_buffer[i]] = idata[i];
                }
            }

            // Compute the number of elements remaining after compaction
            int num_element = bool_buffer[n - 1] + scan_buffer[n - 1];

            timer().endCpuTimer();

            // Memory free
            delete[] bool_buffer;
            delete[] scan_buffer;
            return num_element;
        }
    }
}
