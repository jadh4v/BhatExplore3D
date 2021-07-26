#pragma once
#include <atomic>
#include <future>
#include <map>
#include <mutex>
#include <thrust/device_vector.h>
#include <vtkActor.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImplicitFunction.h>
#include <vtkContourFilter.h>
#include <vtkPolyDataMapper.h>
#include "ActiveContourBlock.h"
#include "ThreadSyncObject.h"

template<size_t _Dim, typename _Attrib>
class MultiBlockSync
{
public:
    typedef ActiveContourBlock<_Dim, _Attrib> BlockType;
    typedef vtkSmartPointer<vtkImageData> ImagePtr;
    typedef vtkSmartPointer<vtkImplicitFunction> ImplicitFunction;

    MultiBlockSync(const int numOfBlocks, ImagePtr displayPhi, ImagePtr pinImage, ImagePtr poutImage, std::atomic<bool>& stopFlag);
    virtual ~MultiBlockSync();

    std::atomic<bool>& StopFlag() { return m_StopFlag; }
    void LockPhi() { m_PhiMutex.lock(); }
    //void ReleasePhi() { m_Phi.data->Modified(); m_PhiMutex.unlock(); }
    void PhiModified();// { m_Phi.data->Modified(); }
    std::mutex& PhiMutex() { return m_PhiMutex; }
    void UpdateGlobalHin(const std::vector<float>& H);
    void UpdateGlobalHout(const std::vector<float>& H);
    void GetGlobalHin(thrust::device_vector<float>& hin);
    void GetGlobalHout(thrust::device_vector<float>& hin);
    float GetGlobalMaxE() const { return m_Global.Max_e; }
    void Launch(ImagePtr inputImage, ImagePtr inputMask, ImplicitFunction maskFunction, std::vector<std::future<void>>& futures);

    MacroGetMember(int, m_MaxNumberOfBlocks, NumberOfBlocks)
    MacroSetMember(vtkSmartPointer<vtkPolyData>, m_Phi.contour, PhiContour)
    MacroSetMember(vtkSmartPointer<vtkContourFilter>, m_Phi.contourFilter, PhiContourFilter)
    MacroSetMember(vtkSmartPointer<vtkPolyDataMapper>, m_Phi.mapper, PhiContourMapper)
    MacroSetMember(vtkSmartPointer<vtkActor>, m_Phi.actor, PhiContourActor)
    //void SetPhiContour(vtkSmartPointer<vtkPolyData> in)
    //{
    //    m_Phi.contour = in;
    //}

    //void HistogramHostSync();
    //void HistogramDeviceSync();
    //void GPUMemcpySync();
    void ACBlockSync(unsigned int lineNumber);

    void ClearGlobalHistograms();
    const void* GetLeftLayer(int blockId) const;
    const void* GetRightLayer(int blockId) const;
    int GetGPUId(int blockId) const;
    const void* GetHinDevicePtr(int blockId) const;
    const void* GetHoutDevicePtr(int blockId) const;
    static ImagePtr ClipData(ImagePtr data, int extent[6]);
    void test_Histograms() const;
    void Compute_Global_Max_e();

    bool FutureIsReady();
    struct {
        std::mutex mutex;
        std::future<void> future;
    }m_Contour;

private:

    void _Launch_AC_Block(
        int maxGPUCount,
        int blockId,
        typename ActiveContourBlock<_Dim,_Attrib>::Type type,
        ImagePtr inputImage,
        ImagePtr inputMask,
        float* displayPhi,
        ImagePtr pinImage,
        ImagePtr poutImage,
        ImplicitFunction maskFunction);

    const BlockType* FetchBlock(int blockId) const;

    static typename ActiveContourBlock<_Dim, _Attrib>::Type determine_type(int blockId, int maxBlockId);

    struct {
        std::mutex HinMutex, HoutMutex;
        std::vector<float> Hin;
        std::vector<float> Hout;
        float Max_e = 0.0f;
    } m_Global;

    const int cGhostLayers = 2;
    const int m_MaxNumberOfBlocks = 1;
    std::mutex m_PhiMutex;
    std::atomic<bool>& m_StopFlag;

    // Display elements for active contour
    struct {
        ImagePtr data = nullptr;
        vtkSmartPointer<vtkPolyData> contour = nullptr;
        vtkSmartPointer<vtkContourFilter> contourFilter = nullptr;
        vtkSmartPointer<vtkPolyDataMapper> mapper = nullptr;
        vtkSmartPointer<vtkActor> actor = nullptr;
    }m_Phi;

    ImagePtr m_PinImage = nullptr;
    ImagePtr m_PoutImage = nullptr;

    struct {
        std::mutex mtx;
        std::map<int, const BlockType*> vec;
    } m_Blocks;

    struct {
        std::mutex Mtx;
        std::map<unsigned int, ThreadSyncObject*> LineToSync;
    }m_ACBlockSyncObjects;

    //ThreadSyncObject m_HistogramHostSync, m_HistogramDeviceSync, m_MemcpySync;
};

