#include <vector>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include <thrust/device_vector.h>
#include "bhat/Utils.h"
 
// point grid, planes --> distance function (phi)
__global__ void CudaKernel_ConstructPhi(
    float* phi,
    const float3* points,
    const size_t arraySize,
    const float3* normals,
    const float3* origins,
    const size_t planeCount,
    const uint3 extents)
{ 
    // Calculate normalized texture coordinates
    int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t outIndex = idx.z*extents.x*extents.y + idx.y * extents.x + idx.x;
    if (outIndex >= arraySize)
        return;

    float max_value = -1e10;
    const float3 p = points[outIndex];

    for (size_t i = 0; i < planeCount; ++i)
    {
        const float3 n = normals[i];
        const float3 o = origins[i];
        float value = n.x * (p.x - o.x) + n.y * (p.y - o.y) + n.z * (p.z - o.z);
        max_value = max(value, max_value);
    }
    phi[outIndex] = max_value;
}

void ConstructPhi_CUDA_C(
    std::vector<float>& oPhi,
    const std::vector<float3>& points,
    const size_t numOfPlanes,
    const std::vector<float3>& normals,
    const std::vector<float3>& origins,
    const uint3 extents)
{
    checkCudaErrors(cudaDeviceSynchronize());
    
    thrust::device_vector<float>  d_phi = oPhi;
    thrust::device_vector<float3> d_normals = normals;
    thrust::device_vector<float3> d_origins = origins;
    thrust::device_vector<float3> d_points  = points;

    dim3 dimBlock(8, 8, 8);
    if (extents.z == 1)
        dimBlock.z = 1;

    dim3 dimGrid((uint(extents.x) + dimBlock.x - 1) / dimBlock.x,
        (uint(extents.y) + dimBlock.y - 1) / dimBlock.y,
        (uint(extents.z)  + dimBlock.z - 1) / dimBlock.z);

    CudaKernel_ConstructPhi <<<dimGrid, dimBlock>>>(
        THRUST_2_RAW(d_phi),
        THRUST_2_RAW(d_points),
        d_phi.size(),
        THRUST_2_RAW(d_normals),
        THRUST_2_RAW(d_origins),
        numOfPlanes,
        extents);

    checkCudaErrors(cudaDeviceSynchronize());
    thrust::host_vector<float> tmp = d_phi;
    thrust::copy(tmp.begin(), tmp.end(), oPhi.begin());
    checkCudaErrors(cudaDeviceSynchronize());
}
