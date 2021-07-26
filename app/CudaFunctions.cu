#include "CudaFunctions.h"

namespace CudaFunctions
{
    bool Compare(const thrust::device_vector<float>& vector1, const thrust::device_vector<float>& vector2)
    {
        return thrust::equal(vector1.begin(), vector1.end(), vector2.begin());
    }
}
