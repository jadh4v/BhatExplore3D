#pragma once
#include <thrust/device_vector.h>

namespace CudaFunctions
{
    bool Compare(const thrust::device_vector<float>& vector1, const thrust::device_vector<float>& vector2);
};
