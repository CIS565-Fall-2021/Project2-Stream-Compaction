#include "radix_sort.h"

// Block size used for CUDA kernel launch
#define BLOCK_SIZE 128

namespace Sort {
    // Map each element in idata to 0/1 contrary to its d-th bit
    __global__ void kern_map_bit_to_bool(int n, int d, int *rbools, const int *idata) {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= n) {
            return;
        }

        rbools[index] = !((idata[index] >> d) & 1);
    }

    // Generate the indices of split result for elements with true keys
    __global__ void kern_gen_true_key_index(int n, int falses, int *odata, const int *scan) {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= n) {
            return;
        }

        odata[index] = index - scan[index] + falses;
    }

    // Generate the indices of split result for all elements
    __global__ void kern_gen_index(int n, int *odata, const int *rbools, const int *scan, const int *t) {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= n) {
            return;
        }

        odata[index] = rbools[index] ? scan[index] : t[index];
    }

    // Scatter based on index array 
    __global__ void kern_scatter(int n, int *odata, const int *addr, const int *idata) {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= n) {
            return;
        }

        odata[addr[index]] = idata[index];
    }

    /*
    * Performs split on idata at turn d, storing the result into odata.
    * Output array with false keys before true keys.
    *
    * @param n      The number of elements in idata.
    * @param idata  The array of elements to split.
    * @param rbools True/False for bit d (reversed).
    * @param odata  The result array.
    */
    void split(int n, int *odata, const int *idata, const int *rbools) {
        // Create device array
        int *dev_scan_buffer;
        int *dev_true_buffer;
        int *dev_index_buffer;
        cudaMalloc((void **)&dev_scan_buffer, n * sizeof(int));
        cudaMalloc((void **)&dev_true_buffer, n * sizeof(int));
        cudaMalloc((void **)&dev_index_buffer, n * sizeof(int));
        checkCUDAError("cudaMalloc failed!");

        // Exclusive scan on reversed bool array
        StreamCompaction::Efficient_Upgraded::scan(n, dev_scan_buffer, rbools, false);

        // Used for computing the number of elements remaining after compaction
        int *last_elements = new int[2];

        // Fetch last element of reversed bool array and scan array respectively
        cudaMemcpy(last_elements, rbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(last_elements + 1, dev_scan_buffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy back failed!");

        // Compute the number of total falses
        int total_falses = last_elements[0] + last_elements[1];
        free(last_elements);

        // Generate index array for writing true keys
        dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kern_gen_true_key_index << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, total_falses, dev_true_buffer, dev_scan_buffer);
        checkCUDAError("kern_gen_true_key_index failed!");

        // Generate index array for writing all keys
        kern_gen_index << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_index_buffer, rbools, dev_scan_buffer, dev_true_buffer);
        checkCUDAError("kern_gen_index failed!");

        // Scatter to output
        kern_scatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, odata, dev_index_buffer, idata);
        checkCUDAError("kern_scatter failed!");

        // Cleanup
        cudaFree(dev_scan_buffer);
        cudaFree(dev_true_buffer);
        cudaFree(dev_index_buffer);
        checkCUDAError("cudaFree failed!");
    }

    /*
    * Performs radix sort on idata, storing the result into odata.
    * Sort data from smaller to larger.
    *
    * @param n          The number of elements in idata.
    * @param num_bits   The maximum number of bits.
    * @param idata      The array of elements to sort.
    * @param odata      The result array.
    */
    void radix_sort(int n, int num_bits, int *odata, const int *idata) {
        // Create device array
        int *dev_array;
        int *dev_res;
        int *dev_bool_buffer;
        cudaMalloc((void **)&dev_array, n * sizeof(int));
        cudaMalloc((void **)&dev_res, n * sizeof(int));
        cudaMalloc((void **)&dev_bool_buffer, n * sizeof(int));
        checkCUDAError("cudaMalloc failed!");

        // Copy data to GPU
        cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy failed!");

        dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Split for num_bits times
        for (int k = 0; k < num_bits; k++) {
            // Generate bool array
            kern_map_bit_to_bool << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, k, dev_bool_buffer, dev_array);
            checkCUDAError("kern_map_bit_to_bool failed!");

            split(n, dev_res, dev_array, dev_bool_buffer);

            // Ping-pong the buffers
            cudaMemcpy(dev_array, dev_res, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            checkCUDAError("ping-pong failed!");
        }

        // Copy data back
        cudaMemcpy(odata, dev_res, sizeof(int) * n, cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy back failed!");

        // Cleanup
        cudaFree(dev_array);
        cudaFree(dev_res);
        cudaFree(dev_bool_buffer);
        checkCUDAError("cudaFree failed!");
    }
}