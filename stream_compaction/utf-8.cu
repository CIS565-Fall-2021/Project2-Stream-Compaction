#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "cVec.h"

#define block_size 128

namespace UTF8 {

	/* Format of UTF-8:
	 * Each unicode code-point is encoded with one to four bytes, whose forms are:
	 * 1	0xxxxxxx					U+0000  to  U+007F
	 * 2	110xxxxx 10xxxxxx				U+0080  to  U+07FF
	 * 3	1110xxxx 10xxxxxx 10xxxxxx			U+0800  to  U+FFFF
	 * 4	11110xxx 10xxxxxx 10xxxxxx 10xxxxxx		U+10000 to  U+10FFFF
	 * UTF-8 is a superset of ASCII, such that code-points of length 1 correspond
	 * precisely to ASCII characters.
	 * Valid UTF-8 must be a sequence of these code-points, all other bytes (or
	 * out-of-place bytes) are invalid. Code points above U+10FFFF, even if they
	 * can be represented by the representation above are invalid as of RFC3629.
	 * 
	 * When decoding, the "x" bytes are concatenated into a rune of 4 bytes.
	 * For example, the UTF-8 character 0xEFBFBD (U+FFFD) is converted to 0xFFFD.
	 * Encoding runes to UTF-8 reverses this operation.
	 */

11101111 10111111 10111101
1111 1111 1111 1101

typedef uint8_t byte;
typedef uint32_t rune

	/* decodes a string 'in' of UTF-8 characters of length n bytes, storing the output in 'out'
	 * 'out' must be large enough to contain the output
	 * 'in' must be a valid UTF-8 string
	 * returns the number of code-points decoded
	 */
	size_t cpu_decode(const byte *in, size_t n, rune *out)
	{
		const byte *s;
		int count;

		for (s = in, count = 0; s < in + n; count++, out++) {
			if ((*s & 0x80) == 0) { /* 1-byte code-point */
				*out = (rune) s[0];
				s++:
			} else if ((*s & 0xe0) == 0xc0) { /* 2-byte code-point*/
				*out = ((rune) (s[0] & 0x1f) << 6) | /* bytes from 110xxxxx */
				       ((rune) (s[1] & 0x3f));       /* bytes from 10xxxxxx */
				s += 2;
			} else if ((*s & 0xf0) == 0xe0) { /* 3-byte code-point */
				*out = ((rune) (s[0] & 0x0f) << 12) | /* bytes from 1100xxxx */
				       ((rune) (s[1] & 0x3f) <<  6) | /* bytes from 10xxxxxx */
				       ((rune) (s[2] & 0x3f));        /* bytes from 10xxxxxx */
				s += 3;
			} else {
				/* this implementation assumes the encoding is valid and performs no checks here */
				*out = ((rune) (s[0] & 0x07) << 18) | /* bytes from 11110xxxx */
				       ((rune) (s[1] & 0x3f) << 12) | /* bytes from 10xxxxxxx */
				       ((rune) (s[2] & 0x3f) <<  6) | /* bytes from 10xxxxxxx */
				       ((rune) (s[3] & 0x3f));        /* bytes from 10xxxxxxx */
				s += 4;
			}
		}
		return count;
	}

	/* encodes a list of n code-points from 'in', storing the output in 'out'
	 * 'out' must be large enough to contain the output
	 * 'in' must be a sequence of valid UTF-8 code-points (U+0000 to U+10FFFF or 0x0000 to 0x10FFFF)
	 * returns the number of bytes written to output
	 */
	size_t cpu_encode(const rune *in, size_t n, byte *out)
	{
		const rune *s;
		int count;

		for (s = in, count = 0; s < in + n; s++, count++) {
			if (s < 0x80) { /* 1-byte code point */
				*out++ = (byte) (*s & 0x8f);
			} else if (s < 0x800) {
				*out++ = (byte) (((*s >> 6) & 0x1f) | 0xc0); /* bytes for 110xxxxx */
				*out++ = (byte) (*s & 0x3f) | 0x80;          /* bytes for 10xxxxxx */
			} else if (s < 0x10000) {
				*out++ = (byte) (((*s >> 12) & 0x3f) | 0x0); /* TODO: these sections are incorrect */
			}
		}
		
	}

	/* decodes a string 'in' of UTF-8 characters of length n, storing the output in out
	 * out must be large enough to contain the output (if unknown, 4*n)
	 * parallelizes on the GPU
	 * 'in' must be a valid UTF-8 string
	 */
	void decode(const byte *in, size_t n, rune *out)
	{
			
	}
}
