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

            int N = imakepower2(n);

            thrust::device_vector<int> dv_in(idata, idata + N);
            thrust::device_vector<int> dv_out(N);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::inclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

            thrust::host_vector<int> h_out = dv_out;

            // TODO: not sure why I can't just do odata = &h_out[0];
            for (int i = 0; i < N; i++)
              odata[i] = h_out[i];
        }
    }
}
