#ifndef FEATURESPACECUDA_H
#define FEATURESPACECUDA_H

#include <vector>
#include <thrust/device_vector.h>
#include "CudaTexture.h"

class BhattParameters;

namespace Bhat {

template<size_t _Dim=1, typename _Attrib=float>
class FeatureSpaceCuda
{
public:
    /// Construct an empty feature space.
    FeatureSpaceCuda();
    virtual ~FeatureSpaceCuda();

    /// Construct feature space from given points / point-features.
    FeatureSpaceCuda(
        const unsigned int GPU_device_id,
        const std::vector<_Attrib>& voxel_features,
        const int inputDim[3],
        const BhattParameters& _param );

    void OverridePin();
    void OverridePout();
    void SetPinOverride(const std::vector<float>& _Pin);
    void SetPoutOverride(const std::vector<float>& _Pout);
    std::vector<float> GetPin()  const;
    std::vector<float> GetPout() const;
    std::vector<float> GetHin()  const;
    std::vector<float> GetHout() const;
    thrust::device_vector<float>& Hin() { return m_Hin; }
    thrust::device_vector<float>& Hout() { return m_Hout; }

    void SetExclusionCount(size_t count) { m_exclusionCount = count; }

    void GetPhi(std::vector<float>& _phi) const;
    void GetDisplayHin(std::vector<float>& buffer)  const;
    void GetDisplayHout(std::vector<float>& buffer)  const;
    void GetV(std::vector<float>& V) const;
    bool Stopped() const { return m_Stopped; }

    void SetPhi(const std::vector<float>& _phi);
    void SetBlanked(const std::vector<char>& blanked);
    void Sussman();

    const thrust::device_vector<float>& GetHinDevice()  const { return m_Hin;  }
    const thrust::device_vector<float>& GetHoutDevice() const { return m_Hout; }
    const thrust::host_vector<float>& GetHin_Host()  const { return m_Hin_host;  }
    const thrust::host_vector<float>& GetHout_Host() const { return m_Hout_host; }

    template<typename T>
    static std::vector<T> Get_StdVector(const thrust::device_vector<T>& Input);
    const std::vector<float> GetHin_Std() const;
    const std::vector<float> GetHout_Std() const;

    void StoreHin_To_Host() { m_Hin_host = m_Hin; }
    void StoreHout_To_Host() { m_Hout_host = m_Hout; }
    void Compute_Max_e();
    float Get_Max_e() const { return m_Max_e; }
    void Set_Max_e(float e) { m_Max_e = e; }

    /// Calculate approximate kernel bandwidth (sigma) for all dimensions of the feature space.
    //void BuildKernel();
    /// Compute variance for a component of feature points.
    //double GetVariance(size_t dimension) const;

    /// Compute Kernel-based density estimation for each of the regions.
    void Compute_H();
    void Update_H(const int blockType);
    void Update_H(const float* Hin, const float* Hout, int srcGPUId);

    /// Compute function L
    void ComputeL();

    /// Compute function V
    void ComputeV();

    /// Compute Normalized Histograms as P
    void Compute_P();

    void ComputeDivergence(const double spacing[3]);

    //void ProcessDivergence(const float* alhpaK_data);
    void ProcessDivergence();
    void UpdateFromLeft(const void* srcPhi, const int srcGPUId);
    void UpdateFromRight(const void* srcPhi, const int srcGPUId);
    const void* GetLeftLayer() const;
    const void* GetRightLayer() const;
    const void* GetHinDevicePtr()  const;
    const void* GetHoutDevicePtr() const;
    void SetGlobalHin (const void* srcHin,  const int srcGPUId);
    void SetGlobalHout(const void* srcHout, const int srcGPUId);
    void _GetDisplayP(std::vector<float>& buffer, const thrust::device_vector<float>& H) const;

private:
    struct Texture{
        cudaArray* cuArray = nullptr;
        cudaExtent extent;
        cudaTextureObject_t texObj = 0;
        void Destroy()
        {
            // Destroy texture object
            cudaDestroyTextureObject(texObj);
            // Free device memory
            cudaFreeArray(cuArray);
        }
    };
    /// Create CUDA texture object for processing in kernels.
    //Texture _AllocateTexture(cudaChannelFormatDesc& channelDesc, cudaExtent& extent);
    /// Validate if passed dim is valid.
    bool _ValidDimension(size_t d) const;
    /// Compute second term of V(x)
    double _Compute_V_1st_Term() const;

    //std::vector<_Attrib> _Dirac(_Attrib x, _Attrib sigma);
    inline size_t _SizeOfZSlice() const { return m_extent.width * m_extent.height; }
    inline size_t _GhostSize() const { return _SizeOfZSlice()*cGhostLayers; }

    //void _Compute_A();
    void _Compute_A_From_H();
    void _Compute_P();
    void _ClearUpdateMarkers();
    size_t _HistSize() const { return (size_t)(pow(m_Param.HistSize(), _Dim)); }
    size_t _HistDimSize() const { return m_Param.HistSize(); }

    // float functions
    void _Histogram(thrust::device_vector<float>& fg, thrust::device_vector<float>& bg);
    void _Convolve(thrust::device_vector<float>& P, int iterations);
    void _Convolve1(thrust::device_vector<float>& P, int iterations);
    //void _Convolve2(thrust::device_vector<float>& P, int iterations);
    void Convolve_CUDA(thrust::device_vector<float>& P, int iterations);

    // float2 functions
    void _Histogram(thrust::device_vector<float2>& fg, thrust::device_vector<float2>& bg);
    void _Histogram(thrust::device_vector<float3>& fg, thrust::device_vector<float3>& bg);
    static dim3 GetBlockDimensions(const cudaExtent& extent, const dim3& threads);

    void TestHistogramUpdateWithCPU(std::vector<float>& Hin, std::vector<float>& Hout);

    template<typename T>
    void _SetBlanking(const thrust::device_vector<T>& input, thrust::device_vector<T>& output);

    /// normalize the feature space.
    //void _Normalize(std::vector<Point>& points);
    /// Compute the kernel function value based on individual dimension bandwidths, given (z - z_i).
    //double _Kernel(Point d) const;

    BhattParameters m_Param;
    const double eps = 1e-16;
    thrust::device_vector<float> m_phi, m_included_phi;
    thrust::device_vector<_Attrib> m_points, m_included_points;    /// points in feature space

    bool m_Stopped = false;
    float m_Ain = 0, m_Aout = 0, m_Max_e = 0;
    thrust::host_vector<float> m_Hin_host, m_Hout_host;
    thrust::device_vector<float> m_Pout, m_Pin, m_L, m_V;
    thrust::device_vector<float> m_Hout, m_Hin, m_Hout_bkp, m_Hin_bkp, m_Peer_Hin, m_Peer_Hout;
    thrust::device_vector<signed char> m_updated;
    thrust::device_vector<char> m_blanked; // mask for inclusion and exclusion of voxels for enabling hierarchical execution.
	thrust::device_vector<float> m_buffer;
    size_t m_exclusionCount = 0;
    struct {
        struct {
            bool Enable = false;
            thrust::device_vector<float> data;
        } Pin, Pout;
    }m_Override;

    cudaExtent m_extent;
    CudaTexture<float>* m_TexPhi = 0;
    CudaTexture<float4>* m_TexGrad = 0;
    CudaTexture<float>* m_Histogram = 0;
    const int m_GPUDeviceId;
    int m_GPUCount = 0;
    const int cGhostLayers = 2;
};

}

#endif // FEATURESPACECUDA_H
