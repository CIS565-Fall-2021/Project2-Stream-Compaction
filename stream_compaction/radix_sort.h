#pragma once

#include <cuda.h>
#include "common.h"
#include "efficient.h"

namespace Sort {
    void split(int n, int *odata, const int *idata, const int *rbools);
    void radix_sort(int n, int num_bits, int *odata, const int *idata);
}