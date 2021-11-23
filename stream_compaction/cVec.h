#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

/* smart container for manually allocated CUDA memory, holds both single vars and arrays */

template <typename T>
class cVec {
	size_t n;
	T *data;

	/* resizes the memory to n if different, invalidating the memory contents */
	void resize(int n) {
		if (this->n != n) {
			cudaFree(data);
			checkCUDAError("cVec: cudaFree resize failed!");
			this->n = n;
			cudaMalloc((void **) &data, sizeof(*data) * n);
			checkCUDAError("cVec: cudaMalloc resize failed!");
		}
	}

public:
	constexpr cVec() : n(0), data(nullptr) {}

	/* constructor, takes in host pointer, length of pointer array, and desired capacity of device array
	 * cap must be longer than length */
	cVec(size_t cap, size_t length, const T *host_ptr) : n(length) {
		cudaMalloc((void **) &data, sizeof(*data) * cap);
		checkCUDAError("cVec: cudaMalloc host constructor failed!");
		cudaMemcpy(data, host_ptr, n * sizeof(*data), cudaMemcpyHostToDevice);
		checkCUDAError("cVec: cudaMemcpy host constructor failed!");
	}


	/* constructor, does not zero memory */
	cVec(size_t n) : n(n) {
		cudaMalloc((void **) &data, sizeof(*data) * n);
		checkCUDAError("cVec: cudaMalloc constructor failed!");
	}

	/* copy constructor */
	cVec(const cVec<T>& v) : n(v.n) {  
		cudaMalloc((void **) &data, sizeof(*data) * n);
		checkCUDAError("cVec: cudaMalloc copy constructor failed!");
		cudaMemcpy(data, v.data, n * sizeof(*data), cudaMemcpyDeviceToDevice);
		checkCUDAError("cVec: cudaMemcpy copy constructor failed!");
	}

	/* copy assignment */
	cVec<T>& operator=(const cVec<T>& v) {
		n = 0;
		data = nullptr;
		resize(v.n);
		cudaMemcpy(data, v.data, n * sizeof(*data), cudaMemcpyDeviceToDevice);
		checkCUDAError("cVec: cudaMemcpy copy assignment failed!");
	}

	/* destructor */
	~cVec() {
		cudaFree(data);
		checkCUDAError("cudaFree destructor failed!");
		n = 0;
		data = nullptr;
	}

	/* move constructor */
	explicit cVec(cVec&& v) : n(v.n), data(v.data) {
		v.n = 0;
		v.data = nullptr;
	}

	/* move assignment */
	cVec& operator=(cVec&& v){
		cudaFree(data);
		checkCUDAError("cVec: cudaFree move assignment failed!");
		n = v.n;
		data = v.data;
		v.n = 0;
		v.data = nullptr;
		return *this;
	}

	size_t length() {
		return n;
	}

	size_t size() {
		return n * sizeof(*data);
	}

	T* raw_data() {
		return data;
	}

	const T operator[](size_t i) const {
		T t;
		cudaMemcpy(&t, data + i, sizeof(*data), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy of single byte device to host failed");
		return t;
	}


	/* sets the contents to the array pointed by host_ptr
	 * calling cudaFree and cudaMalloc if necessary
	 */
	cVec& set_from_host(size_t n, const T* host_ptr) {
		resize(n);
		cudaMemcpy(data, host_ptr, n * sizeof(*data), cudaMemcpyHostToDevice);
		checkCUDAError("cVec: cudaMemcpy set_from_host failed!");
		return *this;
	}


	/* copies device memory to main memory and returns a pointer to it
	 * length of returned array is also given by length().
	 * returned array must be manually deleted with delete[]
	 */
	T *get_to_host() {
		T *host_data = new int[n];
		cudaMemcpy(host_data, data, n * sizeof(*data), cudaMemcpyDeviceToHost);
		checkCUDAError("cVec: cudaMemcpy set assignment failed!");
		return host_data;
	}

	/* copies size elements of device memory starting at offset bytes to the given buffer in host memory */
	void copy_to_host(size_t offset, size_t n, T* host_data) {
		cudaMemcpy(host_data, data + offset, n * sizeof(*data), cudaMemcpyDeviceToHost);
		checkCUDAError("cVec: cudaMemcpy copy_to_host failed!");
	}

	/* sets n elements, starting at position offset, to val*/
	cVec& memset(size_t offset, size_t n, T val) {
		cudaMemset(data + offset, val, n * sizeof(*data));
		return *this;
	}

	/* copies n elements from v, starting at offset*/
	void copy_to_range(size_t offset, size_t n, cVec<T> v) {
		cudaMemcpy(data+offset, v.data, n * sizeof(*data), cudaMemcpyDeviceToDevice);
	}

};
