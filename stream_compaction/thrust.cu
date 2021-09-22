#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // Create host vector
            thrust::host_vector<int> input(n);

            // Assign values to host vector
            for (int i = 0; i < n; i++)
            {
                input[i] = idata[i];
            }

            // Create device vectors
            thrust::device_vector<int> d_input = input; // cast from host
            thrust::device_vector<int> d_output(n);

            thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

            // copy output to pointer
            thrust::copy(d_output.begin(), d_output.end(), odata);
            timer().endGpuTimer();
        }
    }
}
