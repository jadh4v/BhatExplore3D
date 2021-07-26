#include <type_traits>
#include <vector>
#include <thread>
#include <cassert>
#include <cuda_runtime_api.h>
#include <QTimer>
#include <QElapsedTimer> 
#include <QFile> 

#include <vtkImageGradient.h>
#include <vtkImageNormalize.h>
#include <vtkImageDivergence.h>
#include <vtkImageMathematics.h>
#include <vtkImageCast.h>
#include <vtkImageGradientMagnitude.h>
#include <vtkImageShiftScale.h>
#include <vtkVariant.h>
#include <vtkPlane.h>
#include <vtkPlanes.h>

#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkRenderer.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageActor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCamera.h>
#include <vtkContourFilter.h>

#include <helper_cuda.h>
#include "bhat/AbstractRegion.h"
#include "bhat/FullRegion.h"
#include "ds/Bitvector.h"
#include "DistField/AlgoCudaDistanceField.h"
#include "utils/utils.h"
#include "AttributeGenerator.h"
#include "MultiBlockSync.h"
#include "ActiveContourBlock.h"

const bool gUseHUpdate = true;
const int host_or_device = 2;

#define HIER_NAN 0
#define MacroTimedCall(proc,print) \
    do { \
        qint64 t = 0;\
        if(print) \
            t = timer.elapsed(); \
        proc; \
        cudaDeviceSynchronize(); \
        if(print) \
            std::cout << #proc << " = " << timer.elapsed() - t << "\n"; \
    } while(0)

#define TEMPL_SIGN template<size_t _Dim, typename _Attrib>
#define CLASS_SIGN ActiveContourBlock<_Dim, _Attrib>
//using template<size_t _Dim, typename _Attrib> TEMPL_SIGN;
//using CLASS_SIGN = ActiveContourBlock<_Dim, _Attrib>;
#define SignMacro(ReturnType) \
    TEMPL_SIGN ReturnType CLASS_SIGN::

typedef unsigned int uint;
using std::cout;
using std::endl;
using Bhat::AttributeGenerator;

// Forward declaration of C style functions that will invoke CUDA kernels.
void ConstructPhi_CUDA_C(
    std::vector<float>& oPhi,
    const std::vector<float3>& points,
    const size_t numOfPlanes,
    const std::vector<float3>& normals,
    const std::vector<float3>& origins,
    const uint3 extents);

TEMPL_SIGN CLASS_SIGN::~ActiveContourBlock()
{
    MacroDelete(m_Space);
}

TEMPL_SIGN
CLASS_SIGN::ActiveContourBlock(
    int maxGPUCount,
    int blockId,
    Type type,
    MultiBlockSync<_Dim,_Attrib>* sync,
    BhattParameters param,
    ImagePtr inputImage,
    ImagePtr inputMask,
    float* displayPhi,
    ImagePtr pinImage,
    ImagePtr poutImage,
    vtkSmartPointer<vtkImplicitFunction> maskFunction)
    : m_Param(param), m_InputMask(inputMask), m_MaskFunction(maskFunction), m_Sync(sync),
    m_BlockId(blockId), m_GPUDevice(blockId % maxGPUCount), m_Type(type)
{
    MacroPrint(m_BlockId);
    m_InputImage = inputImage;
    m_DisplayPhi = displayPhi;
    m_PinImage = pinImage;
    m_PoutImage = poutImage;

#ifdef USE_MATLAB_SUSSMAN
    // Pass vector containing MATLAB data array scalar
    using namespace matlab::engine;
    // Start MATLAB engine synchronously
    m_MatlabPtr = startMATLAB();
#endif
}

TEMPL_SIGN
void CLASS_SIGN::CallMatlab(std::vector<float>& phi, float dt, int dim[3])
{
#if USE_MATLAB_SUSSMAN
    // Create MATLAB data array factory
    matlab::data::ArrayFactory factory;

    matlab::data::ArrayDimensions d{ (size_t)dim[0], (size_t)dim[1], (size_t)dim[2] };
    if (d.back() <= 1)
        d.pop_back();

    matlab::data::TypedArray<float> phi_input = factory.createArray<float>(d, phi.data(), phi.data() + phi.size());

    // Pass vector containing 2 scalar args in vector
    std::vector<matlab::data::Array> args({ phi_input, factory.createScalar<float>(dt) });

    // Call MATLAB function and return result
    matlab::data::TypedArray<float> phi_output = m_MatlabPtr->feval(u"sussman3D", args);
    std::copy(phi_output.begin(), phi_output.end(), phi.begin());
#endif
}

TEMPL_SIGN
void CLASS_SIGN::_MakeNonZero(std::vector<float>& phi, const float zeroValue)
{
    // Use this process to overwrite all exact zero values. We want only non-zero positive or negative values.
    // This helps in checking cases for negative and positive phi later on to identify inside outside.
    for (size_t i = 0; i < phi.size(); ++i)
    {
        float value = phi[i];
        if (fabs(value) < float(1e-10))
            phi[i] = zeroValue;
    }
}

TEMPL_SIGN
void CLASS_SIGN::_ConsoleUpdate(QElapsedTimer& timer, qint64& prevTime) const
{
    qint64 currentTime = timer.elapsed();

	std::cout << "ThreadID: " << std::this_thread::get_id() << " GPU:" << this->m_GPUDevice << " ";
	std::cout << "Iteration time = " << (currentTime - prevTime) / m_Param.PrintIterations() << " ms, ";
	std::cout << "Total time = " << (currentTime) / 1000 << " s \n";

	gLog << "ThreadID: " << std::this_thread::get_id() << " GPU:" << this->m_GPUDevice << " ";
	gLog << "Iteration time = " << (currentTime - prevTime) / m_Param.PrintIterations() << " ms, ";
	gLog << "Total time = " << (currentTime) / 1000 << " s \n";

    prevTime = currentTime;

}

TEMPL_SIGN
void CLASS_SIGN::ConstructMask(vtkImageData* inputMask, const Bhat::AbstractRegion& baseRegion, sjDS::Bitvector& mask)
{
    mask.ClearBits();
    auto data = inputMask->GetPointData()->GetScalars();
    size_t baseSize = baseRegion.Size();
    for (size_t i = 0; i < baseSize; ++i)
    {
        if (data->GetVariantValue(baseRegion[i]).ToDouble() > 0)
        {
            mask.Set(i);
        }
    }
}

TEMPL_SIGN
void CLASS_SIGN::ConstructPhi(const sjDS::Bitvector& mask, const Bhat::AbstractRegion& baseRegion, float* oPhi)
{
    QElapsedTimer timer;
    timer.start();

    // Compute distance field given the mask volume.
    voldef::AlgoCudaDistanceField ds;
    auto ds_prep = timer.elapsed();
    auto domain = baseRegion.Points();
    ds.SetDomainPoints(domain.data(), domain.size());
    auto object = baseRegion.SubRegionBoundaryPoints(mask);
    ds.SetObjectPoints(object.data(), object.size());
    ds.Run();

    std::vector<float> distances = ds.GetOutput();
    auto r = std::minmax_element(distances.begin(), distances.end());
    //cout << "distances { " << *(r.first) << ", " << *(r.second) << "} " << endl;

    // Set negative sign for internal points.
    for (size_t i = 0; i < distances.size(); ++i)
    {
        if (mask.Get(i))
            distances[i] *= -1.0;
    }
    r = std::minmax_element(distances.begin(), distances.end());
    cout << "distances { " << *(r.first) << ", " << *(r.second) << "} " << endl;
    // copy output to a vtkImageData
    std::memcpy(oPhi, distances.data(), sizeof(float)*distances.size());
    //std::cout << "Phi re-compute time = " << timer.elapsed() << " \n";
}

TEMPL_SIGN
void CLASS_SIGN::ConstructPhi(vtkImplicitFunction* func, ImagePtr image, std::vector<float>& oPhi)
{
    vtkIdType sz = image->GetNumberOfPoints();
    for (vtkIdType i = 0; i < sz; ++i)
    {
        double p[3];
        image->GetPoint(i, p);
        oPhi[i] = func->EvaluateFunction(p);
    }
    auto r = std::minmax_element(oPhi.begin(), oPhi.end());
    cout << "distances { " << *(r.first) << ", " << *(r.second) << "} " << endl;
}

TEMPL_SIGN
void CLASS_SIGN::ConstructPhi_CUDA(vtkImplicitFunction* func, ImagePtr image, std::vector<float>& oPhi)
{
    // get the normals and origins of the planes in vector<float> formats
    auto planes = vtkPlanes::SafeDownCast(func);
    int numOfPlanes = planes->GetNumberOfPlanes();

    std::vector<float3> normals;
    std::vector<float3> origins;
    std::vector<float3> points;

    normals.resize(size_t(numOfPlanes));
    origins.resize(size_t(numOfPlanes));
    points.resize(size_t(image->GetNumberOfPoints()));

    for (int i = 0; i < numOfPlanes; ++i)
    {
        double n[3], o[3];
        planes->GetPlane(i)->GetNormal(n);
        planes->GetPlane(i)->GetOrigin(o);
        normals[i] = make_float3(float(n[0]), float(n[1]), float(n[2]));
        origins[i] = make_float3(float(o[0]), float(o[1]), float(o[2]));
    }

    for (int i = 0; i < image->GetNumberOfPoints(); ++i)
    {
        double x[3];
        image->GetPoint(i, x);
        points[i] = make_float3(float(x[0]), float(x[1]), float(x[2]));
    }

    vtkIdType dim[3];
    image->GetDimensions(dim);
    uint3 extents = make_uint3(dim[0], dim[1], dim[2]);

    ConstructPhi_CUDA_C(oPhi, points, (size_t)numOfPlanes, normals, origins, extents);
}

void ReScale(vtkImageData* img, float scalarRange)
{
    double range[2];
    img->GetScalarRange(range);
    std::cout << "Gradient Scalar Range = " << range[0] << ", " << range[1] << std::endl;
    vtkNew<vtkImageShiftScale> shift;
    shift->SetInputData(img);

    shift->SetShift(range[0] * -1.0);
    shift->SetScale(scalarRange/(range[1] - range[0]));
    shift->SetOutputScalarTypeToFloat();
    shift->Update();
    img->DeepCopy(shift->GetOutput());
    img->GetScalarRange(range);
    std::cout << "Gradient Scalar re-scaled Range = " << range[0] << ", " << range[1] << std::endl;
}

TEMPL_SIGN
void CLASS_SIGN::compute_K_div(ImagePtr alphaK, ImagePtr phi, double alpha)
{
    // Compute gradient of phi
    vtkNew<vtkImageGradient> gradientFilter;
    gradientFilter->SetInputData(phi);
    //gradientFilter->SetDimensionality(2);
    gradientFilter->SetNumberOfThreads(m_Param.NumOfThreads());
    gradientFilter->Update();
    // normalize the gradients
    vtkNew<vtkImageNormalize> normalize;
    normalize->SetInputData(gradientFilter->GetOutput());
    normalize->SetNumberOfThreads(m_Param.NumOfThreads());
    normalize->Update();
    auto N = normalize->GetOutput();

/*
    std::cout << "CPU gradient = " << std::endl;
    for (int i = 0; i < N->GetNumberOfPoints(); ++i)
    {
        double g[3];
        N->GetPointData()->GetScalars()->GetTuple(i, g);
        std::cout << "(" << g[0] << ", " << g[1] << ", " << g[2] << ") ";
    }
*/

    // compute divergence of normalized the gradients
    vtkNew<vtkImageDivergence> divFilter;
    divFilter->SetInputData(normalize->GetOutput());
    divFilter->SetNumberOfThreads(m_Param.NumOfThreads());
    divFilter->Update();

    vtkNew<vtkImageMathematics> imageMath;
    imageMath->SetInputData(divFilter->GetOutput());
    imageMath->SetConstantK(alpha);
    imageMath->SetOperationToMultiplyByK();
    imageMath->SetNumberOfThreads(m_Param.NumOfThreads());
    imageMath->Update();

    vtkNew<vtkImageCast> cast;
    cast->SetInputData(imageMath->GetOutput());
    cast->SetOutputScalarTypeToFloat();
    cast->SetNumberOfThreads(m_Param.NumOfThreads());
    cast->Update();

/*
    auto Div = cast->GetOutput();
    std::cout << "CPU divergence = " << std::endl;
    for (int i = 0; i < Div->GetNumberOfPoints(); ++i)
    {
        float value = Div->GetPointData()->GetScalars()->GetVariantValue(i).ToFloat();
        std::cout << " " << value;
    }
*/
    alphaK->DeepCopy(cast->GetOutput());
}

TEMPL_SIGN
void CLASS_SIGN::SetPinOverride(const std::vector<float>& _Pin)
{
    m_Override.Pin.data = _Pin;
    m_Override.Pin.Enable = true;
}

TEMPL_SIGN
void CLASS_SIGN::SetPoutOverride(const std::vector<float>& _Pout)
{
    m_Override.Pout.data = _Pout;
    m_Override.Pout.Enable = true;
}

TEMPL_SIGN
void CLASS_SIGN::GetVoxelAttributes(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<float>& points)
{
    size_t baseSize = baseRegion.Size();
    points.resize(baseSize);

    auto scalars = inputImage->GetPointData()->GetScalars();
    int numOfComps = scalars->GetNumberOfComponents();

    for (size_t i = 0; i < baseSize; ++i)
    {
        if (numOfComps == 1)
        {
            points[i] = (float)scalars->GetComponent(baseRegion[i], 0);
        }
        else if (numOfComps == 2)
        {
            float x = (float)scalars->GetComponent(baseRegion[i], 0);
            float y = (float)scalars->GetComponent(baseRegion[i], 1);
            points[i] = (float)sqrt(x*x + y*y);
        }
        else if (numOfComps == 3)
        {
            float r = (float)scalars->GetComponent(baseRegion[i], 0);
            float g = (float)scalars->GetComponent(baseRegion[i], 1);
            float b = (float)scalars->GetComponent(baseRegion[i], 2);
            int grey = int((11*r + 16*g + 5*b)/32);
            //points[i] = (float)sqrt(x*x + y*y + z*z);
            vtkMath::ClampValue(grey, 0, 255);
            points[i] = (float)grey;
        }
    }
}

TEMPL_SIGN
void CLASS_SIGN::GetVoxelAttributes(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<float2>& points)
{
    MacroPrint(inputImage->GetScalarTypeAsString());
    size_t baseSize = baseRegion.Size();
    points.resize(baseSize);

    auto scalars = inputImage->GetPointData()->GetScalars();
    int numOfComps = scalars->GetNumberOfComponents();

    for (size_t i = 0; i < baseSize; ++i)
    {
        points[i].x = (float)scalars->GetComponent(baseRegion[i],0);
        if (numOfComps > 1)
            points[i].y = (float)scalars->GetComponent(baseRegion[i], 1);
    }

    if (numOfComps == 1)
    {
        vtkNew<vtkImageGradientMagnitude> grads_filter;
        grads_filter->SetInputData(inputImage);
        grads_filter->SetNumberOfThreads(m_Param.NumOfThreads());
        grads_filter->Update();
        MacroPrint(grads_filter->GetOutput()->GetScalarTypeAsString());
        auto grads = grads_filter->GetOutput();
        ReScale(grads, m_Param.ScalarRange());
        auto gradient_scalars = grads->GetPointData()->GetScalars();
        for (size_t i = 0; i < baseSize; ++i)
        {
            double voxelValue = gradient_scalars->GetVariantValue(baseRegion[i]).ToDouble();
            points[i].y = (float)voxelValue;
        }
    }
}

TEMPL_SIGN
void CLASS_SIGN::GetVoxelAttributes(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<float3>& points)
{
    MacroPrint(inputImage->GetScalarTypeAsString());
    size_t baseSize = baseRegion.Size();
    //std::vector<float2> points;
    points.resize(baseSize);

    auto scalars = inputImage->GetPointData()->GetScalars();
    int numOfComps = scalars->GetNumberOfComponents();

    for (size_t i = 0; i < baseSize; ++i)
    {
        points[i].x = (float)scalars->GetComponent(baseRegion[i],0);
        if (numOfComps > 1)
            points[i].y = (float)scalars->GetComponent(baseRegion[i], 1);
        if (numOfComps > 2)
            points[i].z = (float)scalars->GetComponent(baseRegion[i], 2);
    }
}

TEMPL_SIGN
void CLASS_SIGN::_ConstructBlankFlags(std::vector<char>& blanked, ImagePtr inputImage, const size_t sizeOfZSlice)
{
    NullCheck(inputImage);
    //MacroAssert(inputImage->GetScalarType() == VTK_FLOAT);
    vtkIdType sz = inputImage->GetNumberOfPoints();
    blanked.resize(sz);
    const ushort* ushort_scalars = (const ushort*)inputImage->GetScalarPointer();
    const uchar* uchar_scalars = (const uchar*)inputImage->GetScalarPointer();
    const int stride = inputImage->GetNumberOfScalarComponents();
    const int type = inputImage->GetScalarType();
    for (vtkIdType i = 0; i < sz; ++i)
    {
        float value = 0.0f;
        if(type == VTK_UNSIGNED_SHORT)
            value = ushort_scalars[i*stride];
        else
            value = uchar_scalars[i*stride];

        blanked[i] = char(value < 1.0f ? 1 : 0);
    }

#if 1
    // Also blank the ghost voxels.
    // Warning: This can cause a bug in the future if blanked and ghost voxels are treated differently downstream.
    const size_t numOfGhostVoxels = sizeOfZSlice * cGhostLayers;
    if (m_Type == CLASS_SIGN::Intermediate || m_Type == CLASS_SIGN::Last)
    {
        std::fill_n(blanked.begin(), numOfGhostVoxels, char(2));
    }
    if (m_Type == CLASS_SIGN::First || m_Type == CLASS_SIGN::Intermediate)
    {
        std::fill_n(blanked.end()-numOfGhostVoxels, numOfGhostVoxels, char(2));
    }
#endif
}

TEMPL_SIGN
size_t CLASS_SIGN::_CountExcludedVoxels(ImagePtr inputImage)
{
    NullCheck(inputImage,0);
    MacroAssert(inputImage->GetScalarType() == VTK_FLOAT);
    vtkIdType sz = inputImage->GetNumberOfPoints();
    float* scalars = (float*)inputImage->GetScalarPointer();
    int stride = inputImage->GetNumberOfScalarComponents();

    size_t countZeroVoxels = 0;
    for (vtkIdType i = 0; i < sz; ++i)
        if (scalars[i*stride] == 0.0f)
            ++countZeroVoxels;

    return countZeroVoxels;
}

TEMPL_SIGN
void CLASS_SIGN::_UpdateDisplayPhi(SpaceCuda& space, std::vector<float>& phi, const size_t sizeOfZSlice, std::vector<float>& hin, std::vector<float>& hout)
{
    space.GetPhi(phi);
    { // Mutex lock_guard scope
        // update the your own block of display Phi function (there are overlaps with other blocks).
        std::lock_guard<std::mutex> guard(m_Sync->PhiMutex());
        const size_t offset = cGhostLayers * sizeOfZSlice;

        if (m_Type == CLASS_SIGN::First)
        {
            // skip the last z-slice for the first block
            std::copy(phi.begin(), phi.end() - offset, m_DisplayPhi);
        }
        else if (m_Type == CLASS_SIGN::Intermediate)
        {
            std::copy(phi.begin() + offset, phi.end() - offset, m_DisplayPhi);
        }
        else if (m_Type == CLASS_SIGN::Last)
        {
            // skip the first z-slice for non-first (remaining) blocks.
            std::copy(phi.begin() + offset, phi.end(), m_DisplayPhi);
        }
        else if (m_Type == CLASS_SIGN::Single)
        {
            std::copy(phi.begin(), phi.end(), m_DisplayPhi);
        }
        else
        {
            MacroWarning("Invalid block type.");
        }
    }

    m_Sync->ACBlockSync(__LINE__);
    //m_Sync->PhiModified();
    if (m_Type == CLASS_SIGN::Single || m_Type == CLASS_SIGN::First)
    {
        //if (this->_FutureIsReady())
        if(m_Sync->FutureIsReady())
        {
            //std::lock_guard<std::mutex> guard(m_Sync->PhiMutex());
            //this->m_Sync->m_Contour.future = std::async(std::launch::async, [this]() {this->m_Sync->PhiModified(); });
            std::async(std::launch::async, [this]() {this->m_Sync->PhiModified(); });
        }
    }
    //m_Sync->ACBlockSync(__LINE__);

    // temporarily disabled.
    /*
    if (m_BlockId == 0)
    {
        space.GetDisplayHin(hin);
        space.GetDisplayHout(hout);
        std::memcpy(m_PoutImage->GetScalarPointer(), hout.data(), sizeof(float)*hout.size());
        std::memcpy(m_PinImage->GetScalarPointer(), hin.data(), sizeof(float)*hin.size());
        m_PoutImage->Modified();
        m_PinImage->Modified();
    }
    */
}

TEMPL_SIGN
void CLASS_SIGN::_Update_H(QElapsedTimer& timer, const int mode)
{
    m_Space->Update_H(static_cast<int>(m_Type));

    switch(mode)
    {
    case 1:
        _Host_Level_Histogram_Sync(timer);
        break;

    case 2: 
		_GPU_Level_Histogram_Sync(timer);
        break;

    default:
        MacroFatalError("Unrecognized mode.");
        break;
    }
}

TEMPL_SIGN
void CLASS_SIGN::_Compute_H(QElapsedTimer& timer, const int mode)
{
    //MacroMessage("Compute_H");
    m_Space->Compute_H();
    checkCudaErrors(cudaDeviceSynchronize());

    switch(mode)
    {
    case 1:
        _Host_Level_Histogram_Sync(timer);
        break;

    case 2:
        //MacroMessage("_GPU_Level_Histogram_Sync");
        _GPU_Level_Histogram_Sync(timer);
        break;

    default:
        MacroFatalError("Unrecognized mode.");
        break;
    }
}

TEMPL_SIGN
void CLASS_SIGN::_Host_Level_Histogram_Sync(QElapsedTimer& timer)
{
    // send local histogram for global computation
    m_Sync->UpdateGlobalHin(m_Space->GetHin());
    m_Sync->UpdateGlobalHout(m_Space->GetHout());

    // sync all other block threads
    //m_Sync->HistogramHostSync();
    m_Sync->ACBlockSync(__LINE__);

    // get the globally computed histogram
    m_Sync->GetGlobalHin(m_Space->Hin());
    m_Sync->GetGlobalHout(m_Space->Hout());
    m_Space->Compute_P();

    // sync threads (makes sure that all blocks have received their local copy of the Hin and Hout histograms.
    //m_Sync->HistogramHostSync();
    m_Sync->ACBlockSync(__LINE__);

    // clear the global histogram cache for next iteration.
    if(m_BlockId == 0)
        m_Sync->ClearGlobalHistograms();
}

TEMPL_SIGN
void CLASS_SIGN::_GPU_Level_Histogram_Sync(QElapsedTimer& timer)
{
    // sync all other block threads
    //m_Sync->HistogramDeviceSync();
    m_Sync->ACBlockSync(__LINE__);

    // Let the first thread do all the combining of local-histograms (of each block).
    if (m_BlockId == 0)
    {
        int totalNumOfBlocks = m_Sync->GetNumberOfBlocks();
        for (int blockId = 1; blockId < totalNumOfBlocks; ++blockId)
        {
            const float* Hin = (const float*)m_Sync->GetHinDevicePtr(blockId);
            const float* Hout = (const float*)m_Sync->GetHoutDevicePtr(blockId);
            m_Space->Update_H(Hin, Hout, m_Sync->GetGPUId(blockId));
        }
    }

    // sync all other block threads
    //m_Sync->HistogramDeviceSync();
    m_Sync->ACBlockSync(__LINE__);

    if (m_BlockId != 0)
    {
        m_Space->SetGlobalHin(m_Sync->GetHinDevicePtr(0), 0);
        m_Space->SetGlobalHout(m_Sync->GetHoutDevicePtr(0), 0);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    m_Space->Compute_P();

    // sync threads (makes sure that all blocks have received their local copy of the Hin and Hout histograms.
    //m_Sync->HistogramDeviceSync();
    m_Sync->ACBlockSync(__LINE__);
}


TEMPL_SIGN
const void* CLASS_SIGN::GetLeftLayer() const
{
    if (!m_Space)
        return nullptr;

    return m_Space->GetLeftLayer();
}

TEMPL_SIGN
const void* CLASS_SIGN::GetRightLayer() const
{
    if (!m_Space)
        return nullptr;

    return m_Space->GetRightLayer();
}

// Update Phi function in all blocks by exchanging the overlapping z-layers of Phi.
TEMPL_SIGN
void CLASS_SIGN::_UpdatePhiFromOtherBlocks()
{
    // Synchronize Phi with other blocks since the last iteration.
    // blocks are arranged as a chain, where each inner block has a left and a right neighbor.
    m_Sync->ACBlockSync(__LINE__);

    // if not-first-block
    if(m_Type == CLASS_SIGN::Intermediate || m_Type == CLASS_SIGN::Last)
        m_Space->UpdateFromLeft(m_Sync->GetRightLayer(m_BlockId - 1), m_Sync->GetGPUId(m_BlockId - 1));

    // if not-last-block
    //if (m_BlockId < m_Sync->GetNumberOfBlocks() - 1)
    if(m_Type == CLASS_SIGN::First || m_Type == CLASS_SIGN::Intermediate)
        m_Space->UpdateFromRight(m_Sync->GetLeftLayer(m_BlockId + 1), m_Sync->GetGPUId(m_BlockId + 1));

    cudaDeviceSynchronize();
    m_Sync->ACBlockSync(__LINE__);
}

TEMPL_SIGN
const void* CLASS_SIGN::GetHinDevicePtr() const
{
    if (!m_Space)
        return nullptr;

    return m_Space->GetHinDevicePtr();
}

TEMPL_SIGN
const void* CLASS_SIGN::GetHoutDevicePtr() const
{
    if (!m_Space)
        return nullptr;

    return m_Space->GetHoutDevicePtr();
}

TEMPL_SIGN
void CLASS_SIGN::test_Histograms() const
{
#if 0
    m_Space->StoreHin_To_Host();
    m_Space->StoreHout_To_Host();
    this->m_Sync->ACBlockSync(__LINE__);

    if (m_Type == CLASS_SIGN::First)
    {
        this->m_Sync->test_Histograms();
    }
    // Wait for testing thread to finish.
    this->m_Sync->ACBlockSync(__LINE__);
#endif
}
TEMPL_SIGN
void CLASS_SIGN::test_written_Histograms(int stepNumber) const
{
#if 1
    m_Space->StoreHin_To_Host();
    m_Space->StoreHout_To_Host();
    this->m_Sync->ACBlockSync(__LINE__);

    if (m_Type == CLASS_SIGN::First)
    {
        std::vector<float> hin  = m_Space->GetHin_Std();
        std::vector<float> hout = m_Space->GetHout_Std();
        QFile file("histout/hist_" + QString::number(stepNumber));
        MacroOpenQFileToRead(file);
        if (file.isReadable())
        {
            auto bytes = file.readAll();
            //const float* H = (const float*)bytes.constData();
            std::vector<float> H((size_t)bytes.size() / sizeof(float));
            std::copy_n((const float*)bytes.constData(), H.size(), H.data());

            for (size_t i = 0; i < hin.size(); ++i)
                MacroAssert(hin[i] == H[i]);

            for (size_t i = hin.size(); i < hin.size() + hout.size(); ++i)
                MacroAssert(hout[i-hin.size()] == H[i]);
        }
       
        file.close();
    }
    // Wait for testing thread to finish.
    this->m_Sync->ACBlockSync(__LINE__);
#endif
}
TEMPL_SIGN
void CLASS_SIGN::write_Histograms(int stepNumber) const
{
    // write for single GPU mode only
    if (m_Sync->GetNumberOfBlocks() == 1)
    {
        m_Space->StoreHin_To_Host();
        m_Space->StoreHout_To_Host();
        auto hin  = m_Space->GetHin_Std();
        auto hout = m_Space->GetHout_Std();
        QFile file("histout/hist_" + QString::number(stepNumber));
        MacroOpenQFileToWrite(file);
        file.write((const char*)hin.data(), sizeof(float)*hin.size());
        file.write((const char*)hout.data(), sizeof(float)*hout.size());
        file.close();
    }
    /*
    else
    {
        MacroWarning("Should be writing out histogram info for multi-GPU mode.");
    }
    */
}

/// Compute global max_e using all blocks
TEMPL_SIGN
void CLASS_SIGN::_Compute_Global_max_e()
{
    m_Sync->ACBlockSync(__LINE__);

    if (m_Type == CLASS_SIGN::First || m_Type == CLASS_SIGN::Single)
        m_Sync->Compute_Global_Max_e();

    m_Sync->ACBlockSync(__LINE__);
    float global_max_e = m_Sync->GetGlobalMaxE();
    m_Space->Set_Max_e(global_max_e);
}

/*
TEMPL_SIGN
bool CLASS_SIGN::_FutureIsReady() const
{
    bool retValue = m_contourFuture.valid() ? m_contourFuture._Is_ready() : true;
    return retValue;
}
*/


TEMPL_SIGN
void CLASS_SIGN::Run()
{
    // Print Input Information
    double range[2] = { 0,0 }, spacing[3] = { 0,0,0 }, origin[3] = { 0,0,0 };
    int inputDim[3] = { 0,0,0 };
    m_InputImage->GetScalarRange(range);
    m_InputImage->GetDimensions(inputDim);
    m_InputImage->GetSpacing(spacing);
    m_InputImage->GetOrigin(origin);

    std::cout << "input range = " << range[0] << ", " << range[1] << std::endl;
    std::cout << "input dimensions = " << inputDim[0] << ", " << inputDim[1] << ", " << inputDim[2] << std::endl;
    std::cout << "input spacing = " << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << std::endl;
    const size_t sizeOfZSlice = inputDim[0] * inputDim[1];

    m_Param.SetNarrowBand(m_Param.NarrowBand() * double(*std::max_element(spacing, spacing + 3)));
    vtkSmartPointer<vtkImageData> phi_image = Utils::ConstructImage(inputDim, spacing, origin, VTK_FLOAT);

    Bhat::FullRegion baseRegion = Bhat::FullRegion(m_InputImage);
    size_t baseSize = baseRegion.Size();
    std::vector<float> E(baseSize);
    std::vector<float> phi(baseSize);

    std::vector<char> blanked(baseSize);
    //std::fill_n(blanked.begin(), baseSize, false);
    _ConstructBlankFlags(blanked, m_InputImage, sizeOfZSlice);

    sjDS::Bitvector mask(baseSize);
    if(m_MaskFunction)
    {
        MacroMessage("CUDA: vtkPlanes Phi Function.");
        ConstructPhi_CUDA(m_MaskFunction, m_InputImage, phi);
        //ConstructPhi(m_MaskFunction, m_InputImage, phi);
    }
    else if (m_InputMask)
    {
        MacroMessage("CUDA: Distance Phi Function.");
        ConstructMask(m_InputMask, baseRegion, mask);
        ConstructPhi(mask, baseRegion, phi.data());
    }

#if HIER_NAN
    std::transform(phi.begin(), phi.end(), blanked.begin(), phi.begin(), [](auto p, auto h) { return (h ? p : vtkMath::Nan()); });
#endif

    //std::vector<_Attrib> points;
    //GetVoxelAttributes(m_InputImage, baseRegion, points);

    std::vector<_Attrib> points;
    using AttribGen = AttributeGenerator<_Attrib>;
    AttribGen attribGen(m_InputImage, baseRegion, points);
    attribGen.SetGlobalRange(m_Param.GetGlobalRange()[0], m_Param.GetGlobalRange()[1]);
    attribGen.SetHistDimensions(m_Param.HistSize(), m_Param.HistSize(), m_Param.HistSize());
    //const float* ptr = (const float*)m_InputImage->GetScalarPointer();
    if (_Dim == 3)
        attribGen.SetAttributes(AttribGen::AttribRed, AttribGen::AttribGreen, AttribGen::AttribBlue);
        //attribGen.SetAttributes(AttribGen::AttribGrey, AttribGen::AttribGradMag, AttribGen::AttribMedian5);
        //attribGen.SetAttributes(AttribGen::AttribGrey, AttribGen::AttribGrey, AttribGen::AttribGrey);
    else if (_Dim == 2)
        attribGen.SetAttributes(AttribGen::AttribGrey, AttribGen::AttribGradMag);
        //attribGen.SetAttributes(AttribGen::AttribMedian3, AttribGen::AttribGradMag);
        //attribGen.SetAttributes(AttribGen::AttribGrey, AttribGen::AttribGradAngle);
        //attribGen.SetAttributes(AttribGen::AttribRed, AttribGen::AttribBlue);
    else
        attribGen.SetAttributes(AttribGen::AttribGrey);
        //attribGen.SetAttributes(AttribGen::AttribMedian5);
        //attribGen.SetAttributes(AttribGen::AttribRedByBlue);
        //attribGen.SetAttributes(AttribGen::AttribGradMag);

    MacroMessage("Calling AttribGen.Generate()");
    attribGen.Generate();
    m_Space = new SpaceCuda(m_GPUDevice, points, inputDim, m_Param);
    MacroMessage("Space constructed.");
    SpaceCuda& space = *m_Space;
    if (m_Override.Pin.Enable)
        space.SetPinOverride(m_Override.Pin.data);
    if (m_Override.Pout.Enable)
        space.SetPoutOverride(m_Override.Pout.data);

    //m_Space->SetExclusionCount(this->_CountExcludedVoxels(m_InputImage));

    // Use this process to overwrite all exact zero values. We want only non-zero positive or negative values.
    // This helps in checking cases for negative and positive phi later on to identify inside outside.
    //_MakeNonZero(phi, 0.0001f * (*std::min_element(spacing, spacing + 3)));
    space.SetPhi(phi);
    space.SetBlanked(blanked);
    //space.BuildKernel();

    QElapsedTimer timer;
    timer.start();
    qint64 startTime = timer.elapsed();
    qint64 prevTime = 0;
    std::vector<float> hin, hout;

    int steps = 0;
    MacroMessage("Computing H for the first time.");
    try { _Compute_H(timer, host_or_device); }
    catch (std::exception e) { MacroWarning(e.what()); }

    MacroMessage("Done.");
    //test_Histograms();
    //write_Histograms(steps++);
    //test_written_Histograms(steps);

    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);
    assert(gpuCount != 0);
    assert(phi.size() > sizeOfZSlice);
    MacroPrint(space.Stopped());

    // Update Phi function in all blocks by exchanging the overlapping z-layers of Phi.
    _UpdatePhiFromOtherBlocks();

    do
    {
        // Update display objects that update during execution, giving feedback to user.
        // This includes Phi function for drawing active contour surface, and Hin Hout histograms.
        if (steps % 210 == 0)
        {
            //MacroMessage("Updating Phi.");
            if (m_Sync->FutureIsReady())
                _UpdateDisplayPhi(space, phi, sizeOfZSlice, hin, hout);
            //else
                //MacroMessage("Phi update not READY.");
        }

        // Call the Sussman function that enforces a constant slope on the Phi function.
        // This improves convergence but modifies Phi again.
        if (steps % m_Param.RecomputePhiIterations() == 0)
        {

#if USE_MATLAB_SUSSMAN
            space.GetPhi(phi);
            CallMatlab(phi, 0.5f, inputDim);
            space.SetPhi(phi);
#else
            MacroTimedCall(space.Sussman(), 0);
#endif
            _UpdatePhiFromOtherBlocks();

            if (!gUseHUpdate || (gUseHUpdate && (steps % (m_Param.RecomputePhiIterations()*10)) == 0))
                MacroTimedCall(_Compute_H(timer, host_or_device), 0);
            else
                MacroTimedCall(_Update_H(timer, host_or_device), 0);
            //test_Histograms();
        }

        // Print iteration times
        if (steps % m_Param.PrintIterations() == 0)
        {
            if (this->m_Type == CLASS_SIGN::First || this->m_Type == CLASS_SIGN::Single)
                _ConsoleUpdate(timer, prevTime);
        }

        space.OverridePin();
        space.OverridePout();
        MacroTimedCall(space.ComputeL(), 0);
        MacroTimedCall(space.ComputeV(), 0);
        MacroTimedCall(space.ComputeDivergence(spacing), 0);

        space.Compute_Max_e();
        // compute global max_e using all blocks
        _Compute_Global_max_e();

        MacroTimedCall(space.ProcessDivergence(), 0);
        _UpdatePhiFromOtherBlocks();

        //Temp code-begin
#if 0
        if(steps == 20)
        {
            MacroMessage("Saving block phi image.");
            space.GetPhi(phi);
            std::copy(phi.begin(), phi.end(), (float*)phi_image->GetScalarPointer());
            Utils::WriteVolume(phi_image.Get(), QString("phi_image_block_" + QString::number(m_BlockId) + ".mhd").toLatin1().constData());
            Write_V("m_V" + std::to_string(m_BlockId) + ".mhd");
            //exit(0);
        }
        //Temp code-end
#endif

        if (gUseHUpdate)
            MacroTimedCall(_Update_H(timer, host_or_device), 0);
        else
            MacroTimedCall(_Compute_H(timer, host_or_device), 0);

        //test_Histograms();
        //write_Histograms(steps);
        //test_written_Histograms(steps);

        if (space.Stopped())
        {
            MacroPrint(space.Stopped());
            break;
        }
#if 0
        // If stop triggered, do 10 more clean iterations without H update proces.
        if (m_Sync.StopFlag().load())
        {
            m_Sync.StopFlag() = false;
            gUseHUpdate = false;
            steps = m_Param.NumOfIterations() - 50;
        }
#endif
    } while (++steps < m_Param.NumOfIterations() && !m_Sync->StopFlag().load());
    //} while (++steps < m_Param.NumOfIterations())// && !m_Stop.load());
    m_Override.Pin.data = space.GetPin();
    m_Override.Pout.data = space.GetPout();
}

SignMacro(void) Write_V(const std::string& filename) const
{
    std::vector<float> V;
    m_Space->GetV(V);

    sjDS::Grid g;
    g.SetDimensions(m_InputImage->GetDimensions());
    g.SetSpacing(m_InputImage->GetSpacing());
    
    double o[3];
    m_InputImage->GetPoint(0, o);
    g.SetOrigin(o);

    Utils::WriteVolume(V, g, filename);
}




template class ActiveContourBlock<1, float>;
template class ActiveContourBlock<2, float2>;
template class ActiveContourBlock<3, float3>;
