#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
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
            thrust::host_vector<int> hst_odata(odata, odata + n), hst_idata(idata, idata + n);
            thrust::device_vector<int> dev_odata(hst_odata), dev_idata(hst_idata);
            
            timer().startGpuTimer();
            // DONE use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }


        struct IsZero {
            __device__ bool operator()(const int x) {
                return x == 0;
            }
        } isZero;
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
            thrust::host_vector<int> hst_odata(idata, idata + n);
            thrust::device_vector<int> dev_odata(hst_odata);

            timer().startGpuTimer();
            // DONE
            thrust::detail::normal_iterator<thrust::device_ptr<int>> dev_endIt = thrust::remove_if(dev_odata.begin(), dev_odata.end(), isZero);
            timer().endGpuTimer();
            int count = (dev_endIt - dev_odata.begin());
            //cudaMemcpy(odata, dev_odata.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata.data().get(), sizeof(int) * count, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            return count;
            //int count = 0;
            //for (int i = 0; i < n; ++i) {
            //    count += odata[i] == 0 ? 0 : 1;
            //}
            //return count;
        }

        void sort(int n, int* odata, const int* idata) {
            thrust::host_vector<int> hst_idata(idata, idata + n);
            thrust::device_vector<int> dev_idata(hst_idata);

            timer().startGpuTimer();
            thrust::sort(dev_idata.begin(), dev_idata.end());
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
    }
}
