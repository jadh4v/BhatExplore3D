#include "CudaTexture.h"
//#include "bhat/Utils.h"
#include "helper_cuda.h"

#define TEMPLATE_SIGN template<typename VoxelType>
#define CLASS_SIGN CudaTexture<VoxelType>

TEMPLATE_SIGN
CLASS_SIGN::CudaTexture(cudaExtent extent)
{
    m_Extent = extent;
    AllocateTexture();
}

TEMPLATE_SIGN
CLASS_SIGN::CudaTexture(cudaExtent extent, int normalizedCoords)
{
    m_Extent = extent;
    m_NormalizedCoords = normalizedCoords;
    AllocateTexture();
}

TEMPLATE_SIGN
CLASS_SIGN::CudaTexture(cudaExtent extent, int normalizedCoords, cudaChannelFormatDesc desc)
{
    m_Extent = extent;
    m_NormalizedCoords = normalizedCoords;
    m_ChannelDesc = desc;
    AllocateTexture();
}

TEMPLATE_SIGN
CLASS_SIGN::~CudaTexture()
{
    checkCudaErrors(cudaDestroyTextureObject(m_TexObj));
    checkCudaErrors(cudaFreeArray(m_cuArray));
}

TEMPLATE_SIGN
void CLASS_SIGN::AllocateTexture()
{
    // Allocate CUDA array in device memory
    checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &m_ChannelDesc, m_Extent));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = m_NormalizedCoords;

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(&m_TexObj, &resDesc, &texDesc, NULL));
}

TEMPLATE_SIGN
void CLASS_SIGN::LoadData(thrust::device_vector<VoxelType>& data)
{
    if (m_Extent.depth <= 1)
    {
        // Copy for to 1D/2D texture object
        checkCudaErrors(cudaMemcpyToArray(m_cuArray, 0, 0, thrust::raw_pointer_cast(data.data()), data.size() * sizeof(VoxelType), cudaMemcpyDeviceToDevice));
    }
    else
    {
        // Copy for to 3D texture object
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(data.data()), m_Extent.width * sizeof(VoxelType), m_Extent.width, m_Extent.height);
        copyParams.dstArray = m_cuArray;
        copyParams.extent = m_Extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }
}

template class CudaTexture<float>;
template class CudaTexture<float4>;