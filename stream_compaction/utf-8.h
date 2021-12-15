#pragma once

namespace UTF8 {

	/* decodes a string 'in' of UTF-8 characters of length n, storing the output in out
	 * out must be large enough to contain the output (if unknown, 4*n)
	 * 'in' must be a valid UTF-8 string
	 */
	void cpu_decode(const char *in, size_t n, uint8_t *out);

	/* decodes a string 'in' of UTF-8 characters of length n, storing the output in out
	 * out must be large enough to contain the output (if unknown, 4*n)
	 * parallelizes on the GPU
	 * 'in' must be a valid UTF-8 string
	 */
	void decode(const char *in, size_t n, uint8_t *out);
}
