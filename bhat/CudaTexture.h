#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "core/macros.h"

template<typename VoxelType>
class CudaTexture
{
public:
    ~CudaTexture();
    CudaTexture(cudaExtent extent);
    CudaTexture(cudaExtent extent, int normalizedCoords);
    CudaTexture(cudaExtent extent,  int normalizedCoords, cudaChannelFormatDesc desc);
    MacroGetMember(cudaExtent, m_Extent, Extent)
    void LoadData(thrust::device_vector<VoxelType>& data);
    cudaTextureObject_t TexObj() const { return m_TexObj; }

private:
    void AllocateTexture();

    cudaArray* m_cuArray = nullptr;
    cudaExtent m_Extent;
    cudaTextureObject_t m_TexObj = 0;
    int m_NormalizedCoords = 1;
    cudaChannelFormatDesc m_ChannelDesc = cudaCreateChannelDesc<VoxelType>();
};