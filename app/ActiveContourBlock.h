#pragma once
#include <atomic>
#include <future>
#include <qglobal.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImplicitFunction.h>
#include "BhattParameters.h"
#include "FeatureSpaceCuda.h"

//#define USE_MATLAB_SUSSMAN

#ifdef USE_MATLAB_SUSSMAN
    #include "MatlabEngine.hpp"
    #include "MatlabDataArray.hpp"
#endif

class QElapsedTimer;

namespace sjDS {
    class Bitvector;
}
namespace Bhat{
    class AbstractRegion;
}

template<size_t _Dim, typename _Attrib> class MultiBlockSync;

template<size_t _Dim, typename _Attrib>
class ActiveContourBlock
{
public:
    typedef Bhat::FeatureSpaceCuda<_Dim, _Attrib> SpaceCuda;
    typedef vtkSmartPointer<vtkImageData> ImagePtr;
    enum Type{ First = 0, Intermediate, Last, Single };

    virtual ~ActiveContourBlock();

    ActiveContourBlock(
        int maxGPUCount,
        int blockId,
        enum Type type,
        MultiBlockSync<_Dim,_Attrib>* sync,
        BhattParameters param,
        ImagePtr inputImage,
        ImagePtr inputMask,
        float* displayPhi,
        ImagePtr pinImage,
        ImagePtr poutImage,
        vtkSmartPointer<vtkImplicitFunction> maskFunction);

    void Run();
    
    void SetPinOverride(const std::vector<float>& _Pin);
    std::vector<float> GetPin() const { return m_Override.Pin.data; }
    void SetPoutOverride(const std::vector<float>& _Pout);
    std::vector<float> GetPout() const { return m_Override.Pout.data; }
    const void* GetLeftLayer() const;
    const void* GetRightLayer() const;
    const void* GetHinDevicePtr() const;
    const void* GetHoutDevicePtr() const;

    const thrust::device_vector<float>& GetHinDevice()  const { return m_Space->GetHinDevice();  }
    const thrust::device_vector<float>& GetHoutDevice() const { return m_Space->GetHoutDevice(); }

    int GetGPUId() const { return m_GPUDevice; }
    int GetId() const { return m_BlockId; }

    const SpaceCuda* Space() const { return m_Space; }

private:
    void CallMatlab(std::vector<float>& phi, float dt, int dim[3]);
    void ConstructMask(vtkImageData* inputMask, const Bhat::AbstractRegion& baseRegion, sjDS::Bitvector& mask);
    void ConstructPhi(const sjDS::Bitvector& mask, const Bhat::AbstractRegion& baseRegion, float* oPhi);
    void ConstructPhi(vtkImplicitFunction* func, ImagePtr image, std::vector<float>& oPhi);
    void ConstructPhi_CUDA(vtkImplicitFunction* func, ImagePtr image, std::vector<float>& oPhi);
    void compute_K_div(ImagePtr alphaK, ImagePtr phi, double alpha);
    void _ConstructBlankFlags(std::vector<char>& blanked, ImagePtr inputImage, const size_t sizeOfZSlice);
    size_t _CountExcludedVoxels(ImagePtr inputImage);
    void _UpdateDisplayPhi(SpaceCuda& space, std::vector<float>& phi, const size_t sizeOfZSlice, std::vector<float>& hin, std::vector<float>& hout);
    static void _MakeNonZero(std::vector<float>& phi, const float zeroValue);

    void _Update_H(QElapsedTimer& timer, const int mode);
    void _Compute_H(QElapsedTimer& timer, const int mode);
    void _GPU_Level_Histogram_Sync(QElapsedTimer& timer);
    void _Host_Level_Histogram_Sync(QElapsedTimer& timer);
    //bool _FutureIsReady() const;

    /// Update Phi function in all blocks by exchanging the overlapping z-layers of Phi.
    void _UpdatePhiFromOtherBlocks();

    inline void _ConsoleUpdate(QElapsedTimer& timer, qint64& prevTime) const;

    void test_Histograms() const;
    void write_Histograms(int stepNumber) const;
    void test_written_Histograms(int stepNumber) const;

    void _Compute_Global_max_e();
    void Write_V(const std::string& filename) const;


    //============================================
    SpaceCuda* m_Space = nullptr;
    MultiBlockSync<_Dim,_Attrib>* const m_Sync;
    BhattParameters m_Param;
    ImagePtr m_InputImage=0, m_PinImage=0, m_PoutImage=0;
    float* m_DisplayPhi = 0;
    vtkSmartPointer<vtkImageData> m_InputMask = nullptr;
    vtkSmartPointer<vtkImplicitFunction> m_MaskFunction = nullptr;
    const int m_GPUDevice;
    const int m_BlockId;
    const Type m_Type;
    const int cGhostLayers = 2;

#ifdef USE_MATLAB_SUSSMAN
    std::unique_ptr<matlab::engine::MATLABEngine> m_MatlabPtr = nullptr;
#endif

    struct {
        struct {
            bool Enable = false;
            std::vector<float> data;
        } Pin, Pout;
    }m_Override;

    void GetVoxelAttributes(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<float>& points);
    void GetVoxelAttributes(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<float2>& points);
    void GetVoxelAttributes(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<float3>& points);
};

