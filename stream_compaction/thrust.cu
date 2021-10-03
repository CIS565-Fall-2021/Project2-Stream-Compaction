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
            int depth = ilog2ceil(n);
            int array_length = pow(2, depth);
            if (ilog2(n) != depth) {
                int* new_idata = new int[array_length];
                memset(new_idata, 0, array_length * sizeof(int));
                memcpy(new_idata, idata, n * sizeof(int));
                idata = new_idata;
            }
            thrust::host_vector<int> host_idata(idata, idata + array_length);
            thrust::host_vector<int> host_odata(array_length);
            thrust::device_vector<int> dev_idata(array_length);
            thrust::device_vector<int> dev_odata(array_length);
            dev_idata = host_idata;

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();
            host_odata = dev_odata;
            thrust::copy(host_odata.begin(), host_odata.end(), odata);

        }
    }
}
