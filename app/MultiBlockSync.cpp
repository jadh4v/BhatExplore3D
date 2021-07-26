#include <numeric>
#include <vtkImageClip.h>
#include <vtkContourFilter.h>
#include <chrono>
#include <thrust/execution_policy.h>
#include "MultiBlockSync.h"
#include "CudaFunctions.h"

using namespace std::chrono_literals;
//#define DIM 1

extern BhattParameters gParam;
typedef vtkSmartPointer<vtkImageData> ImagePtr;

#define TEMPL_SIGN template<size_t _Dim, typename _Attrib>
#define CLASS_SIGN MultiBlockSync<_Dim, _Attrib>

TEMPL_SIGN
CLASS_SIGN::MultiBlockSync(const int numOfBlocks, ImagePtr displayPhi, ImagePtr pinImage, ImagePtr poutImage, std::atomic<bool>& stopFlag)
    :m_MaxNumberOfBlocks(numOfBlocks), m_PinImage(pinImage), m_PoutImage(poutImage), m_StopFlag(stopFlag)
{
    m_Phi.data = displayPhi;
    MacroAssert(m_Phi.data->GetScalarType() == VTK_FLOAT);
    //m_Phi.contourFilter->SetInputData(m_Phi.data);
}

TEMPL_SIGN
CLASS_SIGN::~MultiBlockSync()
{
    for (auto b : m_Blocks.vec)
    {
        MacroDelete(b.second);
    }
}

TEMPL_SIGN
ImagePtr CLASS_SIGN::ClipData(ImagePtr data, int extent[6])
{
    if (data == nullptr || data->GetNumberOfPoints() == 0)
        return nullptr;

    vtkNew<vtkImageClip> clipper;
    clipper->SetInputData(data);
    clipper->SetOutputWholeExtent(extent);
    clipper->ClipDataOn();
    clipper->Update();
    return clipper->GetOutput();
}

TEMPL_SIGN
typename ActiveContourBlock<_Dim, _Attrib>::Type
CLASS_SIGN::determine_type(int blockId, int maxBlockId)
{
    if (blockId == 0 && maxBlockId == 0)
        return BlockType::Single;
    else if (blockId == 0 && maxBlockId != 0)
        return BlockType::First;
    else if (blockId == maxBlockId)
        return BlockType::Last;
    else
        return BlockType::Intermediate;
}

TEMPL_SIGN
void CLASS_SIGN::Launch(ImagePtr inputImage, ImagePtr inputMask, ImplicitFunction maskFunction, std::vector<std::future<void>>& futures)
{
    // split dataset for parallel GPU processing.
    // Currently we are splitting only along Z-axis for easy programming and efficient memory sync.

    m_StopFlag = false;
    m_Global.Hin.resize(gParam.HistArraySize(_Dim)); // DIM
    m_Global.Hout.resize(gParam.HistArraySize(_Dim)); // DIM

    int gpuCount = 0;
    auto err = cudaGetDeviceCount(&gpuCount);
    if(err != cudaError::cudaSuccess)
        std::cout << "error on cudaGetDeviceCount() = " << err << std::endl;
    else
        std::cout << "Cuda devices available = " << gpuCount << std::endl;

    std::cout << "Parallel Blocks to start = " << m_MaxNumberOfBlocks << std::endl;

    int origExt[6];
    inputImage->GetExtent(origExt);
    // Apply some minimum thickness constraint because we don't want to deal with boundary conditions of slim blocks.
    //MacroAssert(origExt[5] - origExt[4] >= 8*m_MaxNumberOfBlocks);

    // compute the size of a single z-slice in terms of number of voxels.
    int origDim[3];
    inputImage->GetDimensions(origDim);
    const size_t sizeOfZSlice = size_t(origDim[0] * origDim[1]);

    // compute non-overlapping thickness of each block
    int ext[6];
    inputImage->GetExtent(ext);
    const int zThickness = (ext[5] - ext[4] + 1) / m_MaxNumberOfBlocks;

    // compute z-range for first block ( including overlapping zones):
    if(m_MaxNumberOfBlocks > 1)
        ext[5] = ext[4] + zThickness - 1 + cGhostLayers;

    for (int blockId = 0; blockId < m_MaxNumberOfBlocks; ++blockId)
    {
        // create data and mask based on extents of the current block.
        auto data = ClipData(inputImage, ext);
        auto mask = ClipData(inputMask, ext);

        int phi_position[] = { ext[0], ext[2], (blockId == 0 ? ext[4] : ext[4] + cGhostLayers) };
        float* displayPhiPosition = static_cast<float*>(m_Phi.data->GetScalarPointer(phi_position));

        // instantiate thread for current block
        auto type = determine_type(blockId, m_MaxNumberOfBlocks - 1);
        MacroPrint(type);

        auto fut = std::async( std::launch::async, &CLASS_SIGN::_Launch_AC_Block, this,
            gpuCount, blockId, type, data, mask, displayPhiPosition, m_PinImage, m_PoutImage, maskFunction);

        //Secure the returned future, otherwise the main-thread will get blocked when exiting this scope.
        futures.push_back(std::move(fut));

        //Compute z-range for next block (including overlapping zones).
        ext[4] = ext[5] - cGhostLayers - 1;

        // if next block is an intermediate block, we need overlapping voxel-zone on both sides.
        // else if next block is the last block, we need overlap at the z-min and z-max should cover the remaining dataset.
        auto nextType = determine_type(blockId+1, m_MaxNumberOfBlocks - 1);
        if(nextType == BlockType::Intermediate)
            ext[5] = ext[4] + zThickness - 1 + 2*cGhostLayers;
        else 
            ext[5] = origExt[5];
    }
}

TEMPL_SIGN
void CLASS_SIGN::_Launch_AC_Block(
    int maxGPUCount,
    int blockId,
    typename BlockType::Type type,
    ImagePtr inputImage,
    ImagePtr inputMask,
    float* displayPhi,
    ImagePtr pinImage,
    ImagePtr poutImage,
    ImplicitFunction maskFunction)
{
    auto block = new BlockType(maxGPUCount, blockId, type, this, gParam, inputImage, inputMask, displayPhi, pinImage, poutImage, maskFunction);

    {  //lock_guard scope
        std::lock_guard<std::mutex> my_guard(this->m_Blocks.mtx);
        this->m_Blocks.vec.insert(std::make_pair(block->GetId(), block));
    }

    /*
    if (!m_Samples.empty() && DIM == 1)
    {
        // Normalize the samples
        m_Pout.resize(pow(gParam.HistSize(), DIM));
        std::fill(m_Pout.begin(), m_Pout.end(), 0);
        /*
        for (int i = 0; i < (int)m_Pout.size(); ++i)
        {
            m_Pout[i] = std::count(m_Samples.begin(), m_Samples.end(), float(i));
        }
        * /
        for(auto s : m_Samples)
        {
            if (s < m_Pout.size())
                ++m_Pout[size_t(s)];
        }
        auto sum = std::accumulate(m_Pout.begin(), m_Pout.end(), 0, std::plus<float>());
        sum = sum <= 0 ? 1 : sum;
        std::transform(m_Pout.begin(), m_Pout.end(), m_Pout.begin(), [sum](auto& x) { return x / sum; });
    }

    if (m_OverrideEnabled && DIM == 1)
    {
        if (!m_Pout.empty())
            block.SetPoutOverride(m_Pout);
    }
    */

    block->Run();
    /*
    if (m_OverrideEnabled)
    {
        if (m_Pout.empty())
            m_Pout = block.GetPout();
    }
    */
}

TEMPL_SIGN
void CLASS_SIGN::UpdateGlobalHin(const std::vector<float>& localHistogram)
{
    if (m_MaxNumberOfBlocks <= 1) return;
    //thrust::transform(localHistogram.begin(), localHistogram.end(), m_GlobalHistogram.begin(), m_GlobalHistogram.begin(), [](auto h, auto H) {return (h + H); });
    std::lock_guard<std::mutex> lock(m_Global.HinMutex);
    thrust::transform(localHistogram.begin(), localHistogram.end(), m_Global.Hin.begin(), m_Global.Hin.begin(), thrust::plus<float>());
}

TEMPL_SIGN
void CLASS_SIGN::UpdateGlobalHout(const std::vector<float>& localHistogram)
{
    if (m_MaxNumberOfBlocks <= 1) return;
    std::lock_guard<std::mutex> lock(m_Global.HoutMutex);
    thrust::transform(localHistogram.begin(), localHistogram.end(), m_Global.Hout.begin(), m_Global.Hout.begin(), thrust::plus<float>());
}

TEMPL_SIGN
void CLASS_SIGN::GetGlobalHin(thrust::device_vector<float>& h)
{
    if (m_MaxNumberOfBlocks <= 1) return;
    std::lock_guard<std::mutex> lock(m_Global.HinMutex);
    thrust::copy(m_Global.Hin.begin(), m_Global.Hin.end(), h.begin());
    //h = m_Global.Hin;
}

TEMPL_SIGN
void CLASS_SIGN::GetGlobalHout(thrust::device_vector<float>& h)
{
    if (m_MaxNumberOfBlocks <= 1) return;
    std::lock_guard<std::mutex> lock(m_Global.HoutMutex);
    thrust::copy(m_Global.Hout.begin(), m_Global.Hout.end(), h.begin());
    //h = m_Global.Hout;
}

/*
TEMPL_SIGN
void CLASS_SIGN::HistogramHostSync()
{
    //m_HistogramHostSync.Sync();
}

TEMPL_SIGN
void CLASS_SIGN::HistogramDeviceSync()
{
    //m_HistogramDeviceSync.Sync();
}

TEMPL_SIGN
void CLASS_SIGN::GPUMemcpySync()
{
    //m_MemcpySync.Sync();
}
*/

TEMPL_SIGN
void CLASS_SIGN::ClearGlobalHistograms()
{
    if (m_MaxNumberOfBlocks <= 1) return;
    std::lock_guard<std::mutex> lock_hin(m_Global.HinMutex);
    std::lock_guard<std::mutex> lock_hou(m_Global.HoutMutex);
    std::fill( m_Global.Hin.begin(),  m_Global.Hin.end(),  0.0f );
    std::fill( m_Global.Hout.begin(), m_Global.Hout.end(), 0.0f );
}

TEMPL_SIGN
const ActiveContourBlock<_Dim, _Attrib>* CLASS_SIGN::FetchBlock(int blockId) const
{
    auto fnd = m_Blocks.vec.find(blockId);
    if (fnd != m_Blocks.vec.end())
    {
        return fnd->second;
    }
    else
    {
        MacroWarning("Invalid blockId: " << blockId);
        return nullptr;
    }
}

TEMPL_SIGN
const void* CLASS_SIGN::GetLeftLayer(int blockId) const
{
    const BlockType* b = FetchBlock(blockId);
    return  (b == nullptr ? nullptr : b->GetLeftLayer());
}

TEMPL_SIGN
const void* CLASS_SIGN::GetRightLayer(int blockId) const
{
    const BlockType* b = FetchBlock(blockId);
    return (b == nullptr ? nullptr : b->GetRightLayer());
}

TEMPL_SIGN
int CLASS_SIGN::GetGPUId(int blockId) const
{
    const BlockType* b = FetchBlock(blockId);
    return (b == nullptr ? -1 : b->GetGPUId());
}

TEMPL_SIGN
const void* CLASS_SIGN::GetHinDevicePtr(int blockId) const
{
    const BlockType* b = FetchBlock(blockId);
    return (b == nullptr ? nullptr : b->GetHinDevicePtr());
}

TEMPL_SIGN
const void* CLASS_SIGN::GetHoutDevicePtr(int blockId) const
{
    const BlockType* b = FetchBlock(blockId);
    return (b == nullptr ? nullptr : b->GetHoutDevicePtr());
}

TEMPL_SIGN
void CLASS_SIGN::ACBlockSync(unsigned int lineNumber)
{
    ThreadSyncObject* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(this->m_ACBlockSyncObjects.Mtx);
        auto fnd = this->m_ACBlockSyncObjects.LineToSync.find(lineNumber);
        if (fnd == this->m_ACBlockSyncObjects.LineToSync.end())
        {
            ptr = new ThreadSyncObject(m_MaxNumberOfBlocks);
            fnd = m_ACBlockSyncObjects.LineToSync.insert(std::make_pair(lineNumber, ptr)).first;
        }
        ptr = fnd->second;
    }
    ptr->Sync();
}

TEMPL_SIGN
void CLASS_SIGN::test_Histograms() const
{
    if (m_Blocks.vec.empty())
        return;

    const BlockType* first = m_Blocks.vec.at(0);
    for (int blockId=1; blockId < m_MaxNumberOfBlocks; ++blockId)
    {
        const BlockType* curr = m_Blocks.vec.at(blockId);
        const thrust::host_vector<float>& hin_1 = first->Space()->GetHin_Host();
        const thrust::host_vector<float>& hin_2 = curr->Space()->GetHin_Host();
        bool hin_comparison = CudaFunctions::Compare(hin_1, hin_2);
        MacroAssert(hin_comparison);

        const thrust::host_vector<float>& hout_1 = first->Space()->GetHout_Host();
        const thrust::host_vector<float>& hout_2 = curr->Space()->GetHout_Host();
        bool hout_comparison = CudaFunctions::Compare(hout_1, hout_2);
        MacroAssert(hout_comparison);
    }
}

TEMPL_SIGN
void CLASS_SIGN::Compute_Global_Max_e()
{
    int totalNumOfBlocks = this->GetNumberOfBlocks();
#if 0
    std::vector<float> e_values;
    for (int blockId = 0; blockId < totalNumOfBlocks; ++blockId)
    {
        auto block = this->FetchBlock(blockId);
        float e = block->Space()->Get_Max_e();
        e_values.push_back(e);
        //e_values[blockId] = e;
    }
	m_Global.Max_e = *std::max_element(e_values.begin(), e_values.end());
#else
    float e_values[8];
    for (int blockId = 0; blockId < totalNumOfBlocks; ++blockId)
    {
        auto block = this->FetchBlock(blockId);
        float e = block->Space()->Get_Max_e();
        e_values[blockId] = e;
    }
    m_Global.Max_e = *std::max_element(e_values, e_values + totalNumOfBlocks);
#endif
}

TEMPL_SIGN
bool CLASS_SIGN::FutureIsReady()
{
    std::lock_guard<std::mutex> guard(m_Contour.mutex);
    bool retValue = m_Contour.future.valid() ? m_Contour.future._Is_ready() : true;
    return retValue;
}

TEMPL_SIGN
void CLASS_SIGN::PhiModified()
{
    //m_Phi.actor->VisibilityOff();
    // lock_guard
    {
        std::lock_guard<std::mutex> guard(this->m_PhiMutex);
        m_Phi.data->Modified();
        m_Phi.contourFilter->Update();
    }
    //MacroPrint(m_Phi.mapper->GetProgress());
    vtkNew<vtkPolyData> tmp;
    tmp->DeepCopy(m_Phi.contourFilter->GetOutput());
    m_Phi.mapper->SetInputData(tmp);
    //m_Phi.actor->VisibilityOn();
}

template class MultiBlockSync<1, float>;
template class MultiBlockSync<2, float2>;
template class MultiBlockSync<3, float3>;
