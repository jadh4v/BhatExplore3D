//Std
#include <numeric>
#include <algorithm>

// Thrust
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/extrema.h>
#include <vector_types.h>
#include <vector_functions.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

// Proj
#include <core/macros.h>
#include "FeatureSpaceCuda.h"
#include "BhattParameters.h"
#include "CudaTexture.h"
#include "bhat/Utils.h"

using Bhat::FeatureSpaceCuda;
//using Bhat::NGrid;

#define TEMPLATE_SIGN template<size_t _Dim, typename _Attrib>
#define CLASS_SIGN FeatureSpaceCuda<_Dim, _Attrib>

#include "FeatureSpaceCudaKernels.h"
#include "FeatureSpaceCudaFunctors.h"

TEMPLATE_SIGN 
CLASS_SIGN::FeatureSpaceCuda()
    : m_GPUDeviceId(0)
{
    if (_Dim > 3)
        MacroWarning("Feature space dimensions greater than 3 not supported.");
}

inline size_t padding(size_t in)
{
    size_t m8 = in % 8;
    size_t pad = m8 == 0 ? 0 : 8-m8;
    if(in == 1)
        return 1;
    else
        return (in + pad);
}


TEMPLATE_SIGN CLASS_SIGN::
FeatureSpaceCuda(
    const unsigned int GPU_device_id,
    const std::vector<_Attrib>& voxel_features,
    const int inputDim[3],
    const BhattParameters& _param)
    : m_Param(_param), m_GPUDeviceId(GPU_device_id)
{
    #ifndef __CUDACC_EXTENDED_LAMBDA__
        #error "please compile with --expt-extended-lambda"
    #endif

    if (_Dim > 3)
        MacroWarning("Feature space dimensions greater than 3 not supported.");

    //MacroPrint(__CUDA_ARCH__);
    cudaGetDeviceCount(&m_GPUCount);
    MacroAssert((int)m_GPUDeviceId < m_GPUCount);
    cudaSetDevice(m_GPUDeviceId);

    m_points = voxel_features; 

    // Resize operating vectors
    m_V.resize(m_points.size());
    m_phi.resize(m_points.size());
    m_blanked.resize(m_points.size());
    m_updated.resize(m_points.size());

    m_L.resize(_HistSize());
    m_Pin.resize(_HistSize());
    m_Pout.resize(_HistSize());
    m_Hin.resize(_HistSize());
    m_Hout.resize(_HistSize());
    m_Hin_bkp.resize(_HistSize());
    m_Hout_bkp.resize(_HistSize());
    m_Peer_Hin.resize(_HistSize());
    m_Peer_Hout.resize(_HistSize());
    m_buffer.resize(_HistSize());

    m_extent.width = inputDim[0];
    m_extent.height = inputDim[1];
    m_extent.depth = inputDim[2];

    cudaExtent extent = make_cudaExtent(m_extent.width, m_extent.height, m_extent.depth);
    m_TexPhi = new CudaTexture<float>(extent);
    m_TexGrad = new CudaTexture<float4>(extent);

    cudaExtent histExt;
    const size_t hdsz = _HistDimSize();
    switch (_Dim)
    {
    case 1: histExt = make_cudaExtent(hdsz, 1, 1); break;
    case 2: histExt = make_cudaExtent(hdsz, hdsz, 1); break;
    case 3: histExt = make_cudaExtent(hdsz, hdsz, hdsz); break;
    default: MacroFatalError("Wrong number of dimensions: _Dim = " << _Dim);
    }
    m_Histogram = new CudaTexture<float>(histExt, 0);

#if (KERNEL_RADIUS==8)
    std::vector<float> h_Kernel{0.0f, 0.0f, 0.0005f, 0.0032f, 0.0139f, 0.0417f, 0.0916f, 0.1527f, 0.1964f, \
                                    0.1964f, 0.1527f, 0.0916f, 0.0417f, 0.0139f, 0.0032f, 0.0005f, 0.0f};
#else
    std::vector<float> h_Kernel{ 0.1f, 0.2f, 0.4f, 0.2f, 0.1f };
#endif

    checkCudaErrors(cudaMemcpyToSymbol(c_Kernel, h_Kernel.data(), KERNEL_LENGTH * sizeof(float)));

}

TEMPLATE_SIGN CLASS_SIGN::~FeatureSpaceCuda()
{
    MacroDelete(m_TexPhi);
    MacroDelete(m_TexGrad);
    MacroDelete(m_Histogram);
}

TEMPLATE_SIGN
void CLASS_SIGN::ComputeDivergence(const double spacing[3])
{
    const double alpha = m_Param.Alpha();
    //print_vector("Pout before div =", m_Pout);
    dim3 dimBlock(8, 8, 8);
    //dim3 dimBlock(8, 8, 16);
    if (m_extent.depth == 1)
        dimBlock.z = 1;

    dim3 dimGrid((uint(m_extent.width) + dimBlock.x - 1) / dimBlock.x,
        (uint(m_extent.height) + dimBlock.y - 1) / dimBlock.y,
        (uint(m_extent.depth)  + dimBlock.z - 1) / dimBlock.z);

    m_TexPhi->LoadData(m_phi);
    int3 texDim{ (int)m_extent.width, (int)m_extent.height, (int)m_extent.depth };
    const float3 fspacing{ (float)spacing[0], (float)spacing[1], (float)spacing[2] };
    const float fbandwidth = (const float)m_Param.NarrowBand();

    DirectDivergenceKernel <<< dimGrid, dimBlock >>> (THRUST_2_RAW(m_V), m_V.size(), m_TexPhi->TexObj(), texDim, fspacing, fbandwidth, (const float)alpha);
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
bool CLASS_SIGN::_ValidDimension(size_t d) const
{
    return (d < _Dim);
}

#include <iomanip>
// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v, size_t num=0)
{
    typedef typename Vector::value_type T;
    std::cout << "  " << std::setw(20) << name << "  ";
    if (num == 0)
        num = v.size();

    thrust::copy(v.begin(), v.begin()+num, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

// dense histogram using binary search
template <typename Vector1, typename Vector2>
    std::enable_if_t<std::is_same_v<float, Vector1::value_type>> dense_histogram(Vector1& input, Vector2& histogram, size_t histSize)
{
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector2::value_type IndexType; // histogram index type

    if (input.size() < 4)
        return;

    // copy input data (could be skipped if input is allowed to be modified)
    //thrust::device_vector<ValueType> data(input);

    // print the initial data
    //print_vector("initial data", data);

    // sort data to bring equal elements together
    thrust::sort(input.begin(), input.end());
    //data.resize(200);

    // print the sorted data
    //print_vector("sorted data", data);

    // number of histogram bins is equal to the maximum value plus one
    IndexType num_bins = std::ceil(input.back()) + 1;

    // resize histogram storage
    histogram.resize(size_t(num_bins));

    // find the end of each bin of values
    thrust::counting_iterator<IndexType> search_begin(0);
    thrust::upper_bound(input.begin(), input.end(),
        search_begin, search_begin + size_t(num_bins),
        histogram.begin());

    // print the cumulative histogram
    //print_vector("cumulative histogram", histogram);

    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(histogram.begin(), histogram.end(),
        histogram.begin());

    // print the histogram
    //print_vector("histogram", histogram);
    histogram.resize(histSize);
}

template <typename Vector1, typename Vector2>
std::enable_if_t<!std::is_same_v<float, Vector1::value_type>> dense_histogram(Vector1& input, Vector2& histogram, size_t histSize)
{
    MacroFatalError("This function shouldn't be called.");
}

template<typename T>
thrust::device_vector<T> my_conv( const std::vector<T>& kernel, const thrust::device_vector<T>& data, bool normalize)
{
    MacroAssert(kernel.size() % 2 == 1); // kernel should have odd size
    size_t paddingSize = (kernel.size() - 1) / 2;
    std::vector<T> padded_data(data.size() + 2*paddingSize);
    std::vector<T> padded_out(data.size() + 2*paddingSize);

    thrust::copy(data.begin(), data.end(), padded_data.begin() + paddingSize);

    for (size_t i = paddingSize; i < padded_data.size()-paddingSize; ++i)
    {
        padded_out[i] = std::inner_product(kernel.begin(), kernel.end(), padded_data.begin() + i - paddingSize, T(0));
    }

    std::vector<T> out(data.size());
    thrust::copy(padded_out.begin() + paddingSize, padded_out.end() - paddingSize, out.begin());

    if(normalize)
        my_norm(out);

    thrust::device_vector<T> device_out = out;
    return device_out;
}

template<typename T>
void my_norm(std::vector<T>& data)
{
    T sum = std::accumulate(data.begin(), data.end(), T(0));
    std::for_each(data.begin(), data.end(), [&sum](T& value) { value = value / sum; });
}

/*
TEMPLATE_SIGN
void CLASS_SIGN::_Compute_A()
{
    //MacroPrint(m_Stopped);
    m_Ain = (float)thrust::count_if(m_phi.begin(), m_phi.end(), CudaStencil_Negative<float,bool>());
    m_Aout = float(m_points.size()) - m_Ain - float(m_exclusionCount);
    //MacroPrint(m_Ain);
    //MacroPrint(m_Aout);

    // if either inside area or outside area is approaching zero, stop the process.
    const float minSize = 4;
    if (m_Ain < minSize || m_Aout < minSize)
    {
        MacroMessage("Stopping: m_Ain = " << m_Ain << ", m_Aout = " << m_Aout);
        m_Stopped = true;
    }
}
*/

TEMPLATE_SIGN
void CLASS_SIGN::_Compute_A_From_H()
{
    m_Ain  = thrust::reduce(m_Hin.begin(), m_Hin.end(), 0.0f, thrust::plus<float>());
    m_Aout = thrust::reduce(m_Hout.begin(), m_Hout.end(), 0.0f, thrust::plus<float>());
    checkCudaErrors(cudaDeviceSynchronize());
    // if either inside area or outside area is approaching zero, stop the process.
    const float minSize = 4;
    if (m_Ain < minSize || m_Aout < minSize)
    {
        MacroMessage("Stopping: m_Ain = " << m_Ain << ", m_Aout = " << m_Aout);
        m_Stopped = true;
    }
}

__host__ __device__ bool operator<(const float2& a, const float2& b)
{
    return a.x < b.x;
}
__host__ __device__ bool operator<(const float3& a, const float3& b)
{
    return a.x < b.x;
}

TEMPLATE_SIGN
void CLASS_SIGN::Compute_H()
{
    m_Ain = m_Aout = 0;
    thrust::fill(m_Pin.begin(), m_Pin.end(), 0.0f);
    thrust::fill(m_Pout.begin(), m_Pout.end(), 0.0f);
    thrust::fill(m_Hin.begin(), m_Hin.end(), 0.0f);
    thrust::fill(m_Hout.begin(), m_Hout.end(), 0.0f);

    // restrict the phi values to stencil region by setting all voxels outside the stencil to very high positive value.
    // Size of phi and points array has to be the same, hence we need to copy phi into a temp array that correspond to only the included points.
    // Included points are defined by the initial exclusion mask (for hierarchical execution).
    _SetBlanking(m_phi, m_included_phi);
    try {
        thrust::count_if(m_included_phi.begin(), m_included_phi.end(), CudaStencil_Negative<float, bool>());
    }
    catch (std::exception e) {
        MacroWarning(e.what());
    }

    thrust::device_vector<_Attrib> fgValues(m_included_points.size());
    thrust::device_vector<_Attrib> bgValues(m_included_points.size());
    try
    {
        auto fgEnd = thrust::copy_if(m_included_points.begin(), m_included_points.end(), m_included_phi.begin(), fgValues.begin(), CudaStencil_Negative<float, bool>());
        fgValues.resize(fgEnd - fgValues.begin());
        auto bgEnd = thrust::copy_if(m_included_points.begin(), m_included_points.end(), m_included_phi.begin(), bgValues.begin(), CudaStencil_NonNegative<float, bool>());
        bgValues.resize(bgEnd - bgValues.begin()); 
    }
    catch (std::exception e)
    {
        MacroWarning(e.what());
    }
    checkCudaErrors(cudaDeviceSynchronize());
    /*
    auto fg  = Get_StdVector(fgValues);
    auto bg = Get_StdVector(bgValues);
    auto included_points = Get_StdVector(m_included_points);
    auto points = Get_StdVector(m_points);
    */

    _Histogram(fgValues, bgValues);
    //auto v1 = this->GetHin();
    //auto v2 = this->GetHout();

    checkCudaErrors(cudaDeviceSynchronize());
    // reset update markers.
    thrust::copy_n(m_Hin.begin(), m_Hin.size(), m_Hin_bkp.begin());
    thrust::copy_n(m_Hout.begin(), m_Hout.size(), m_Hout_bkp.begin());
    checkCudaErrors(cudaDeviceSynchronize());
    _ClearUpdateMarkers();
}

TEMPLATE_SIGN
void CLASS_SIGN::_Compute_P()
{
    thrust::transform(m_Hin.begin(), m_Hin.end(), m_Pin.begin(), normalize_functor<float>(m_Ain));
    thrust::transform(m_Hout.begin(), m_Hout.end(), m_Pout.begin(), normalize_functor<float>(m_Aout));
    checkCudaErrors(cudaDeviceSynchronize());
    _Convolve(m_Pin, 1);
    _Convolve(m_Pout, 1);
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
void CLASS_SIGN::_Convolve(thrust::device_vector<float>& P, int iterations)
{
	Convolve_CUDA(P, iterations);
	/*
    if (_Dim == 1)
    {
        _Convolve1(P, iterations);
    }
    else
    {
        Convolve_CUDA(P, iterations);
        //_Convolve2(P, iterations);
    }
	*/
}


TEMPLATE_SIGN
void CLASS_SIGN::_Convolve1(thrust::device_vector<float>& P, int iterations)
{
    std::vector<float> binom{ 0.0f, 0.0000f, 0.0005f, 0.0032f, 0.0139f, 0.0417f,
                                0.0916f, 0.1527f, 0.1964f, 0.1964f, 0.1527f, 0.0916f,
                                0.0417f, 0.0139f, 0.0032f, 0.0005f, 0.0000f };

    for(int i=0; i < iterations; ++i)
        P = my_conv(binom, P,  true);
}

TEMPLATE_SIGN
void CLASS_SIGN::Compute_P()
{
    //_Compute_A();
    _Compute_A_From_H();
    _Compute_P();
}

TEMPLATE_SIGN
void CLASS_SIGN::Update_H(const int blockType)
{
    //const size_t sizeOfZSlice = m_extent.width * m_extent.height;
    size_t beginOffset = 0, endOffset = 0;
    if (blockType == 0 || blockType == 1)
        endOffset = _GhostSize();

    if(blockType == 1 || blockType == 2)
        beginOffset = _GhostSize();

    // restore backup copies of histograms for update operation:
    thrust::copy_n(m_Hin_bkp.begin(), m_Hin_bkp.size(), m_Hin.begin());
    thrust::copy_n(m_Hout_bkp.begin(), m_Hout_bkp.size(), m_Hout.begin());

//#define TEST_WITH_CPU
#ifdef TEST_WITH_CPU
    std::vector<float> Hin(_HistSize()), Hout(_HistSize());
    thrust::copy(m_Hin.begin(), m_Hin.end(), Hin.begin());
    thrust::copy(m_Hout.begin(), m_Hout.end(), Hout.begin());
#endif

    // GPU version to update histograms
    // m_points.size has to match the entire block size defined by extents.
    int threadsPerBlock = (int)_HistDimSize();
    int numOfBlocks =(int(m_points.size() - beginOffset - endOffset) + threadsPerBlock - 1) / threadsPerBlock;
    UpdateHistograms<<<numOfBlocks, threadsPerBlock>>> (
        THRUST_2_RAW(m_Hin),
        THRUST_2_RAW(m_Hout),
        THRUST_2_RAW(m_updated) + beginOffset,
        THRUST_2_RAW(m_points)  + beginOffset,
        THRUST_2_RAW(m_blanked) + beginOffset,
        m_points.size() - beginOffset - endOffset,
        _HistDimSize());

    checkCudaErrors(cudaDeviceSynchronize());

#ifdef TEST_WITH_CPU
    TestHistogramUpdateWithCPU(Hin, Hout);
#endif

    //_Compute_A();
    // create a backup copy of histograms (for update operations later).
    thrust::copy_n(m_Hin.begin(), m_Hin.size(), m_Hin_bkp.begin());
    thrust::copy_n(m_Hout.begin(), m_Hout.size(), m_Hout_bkp.begin());
    _ClearUpdateMarkers();
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN void CLASS_SIGN::TestHistogramUpdateWithCPU(std::vector<float>& Hin, std::vector<float>& Hout)
{
    MacroMessage("Calling CPU version.");
    const size_t hsz = _HistSize();
    std::vector<float> H(hsz);
    std::vector<_Attrib> points = device_to_host(m_points);
    for (size_t i = 0; i < m_updated.size(); ++i)
    {
        signed char up = m_updated[i];
        size_t bucket = attrib_to_index(points[i], _HistDimSize());
        if (up == 1)
            ++H[bucket];
        else if (up == 2)
            --H[bucket];
    }
    MacroMessage("CPU updating histograms.");
    for (size_t i = 0; i < hsz; ++i)
    {
        Hin[i]  += H[i];
        Hout[i] -= H[i];
    }
    MacroMessage("Comparing...");
    auto gpu_Hin = device_to_host(m_Hin);
    auto gpu_Hout = device_to_host(m_Hout);
    for (size_t i = 0; i < hsz; ++i)
    {
        MacroAssert(Hin[i] == gpu_Hin[i]);
        MacroAssert(Hout[i] == gpu_Hout[i]);
    }
    MacroMessage("Comparing...done.");
}

TEMPLATE_SIGN std::vector<float> CLASS_SIGN::GetPin() const
{
    return device_to_host(m_Pin);
}
TEMPLATE_SIGN std::vector<float> CLASS_SIGN::GetPout() const
{
    return device_to_host(m_Pout);
}
TEMPLATE_SIGN std::vector<float> CLASS_SIGN::GetHin() const
{
    return device_to_host(m_Hin);
}
TEMPLATE_SIGN std::vector<float> CLASS_SIGN::GetHout() const
{
    return device_to_host(m_Hout);
}

TEMPLATE_SIGN
void CLASS_SIGN::_GetDisplayP(std::vector<float>& buffer, const thrust::device_vector<float>& H) const
{
    if(buffer.size() != H.size())
        buffer.resize(H.size());

    auto tmp = H;
    float maxValuePin  = *thrust::max_element(m_Pin.begin(), m_Pin.end());
    float maxValuePout = *thrust::max_element(m_Pout.begin(), m_Pout.end());
    float maxValue = std::max(maxValuePin, maxValuePout);

    thrust::transform(tmp.begin(), tmp.end(), tmp.begin(), [maxValue]__device__(float& x) { return log10f(1.0f+x/maxValue*255.0f)*200.0f; });
    //thrust::transform(tmp.begin(), tmp.end(), tmp.begin(), []__device__(float& x) { return log10f(x); });
    buffer = device_to_host(tmp);
}
TEMPLATE_SIGN void CLASS_SIGN::GetDisplayHin(std::vector<float>& buffer)  const
{
    _GetDisplayP(buffer, m_Pin);
}
TEMPLATE_SIGN void CLASS_SIGN::GetDisplayHout(std::vector<float>& buffer)  const
{
    _GetDisplayP(buffer, m_Pout);
}

TEMPLATE_SIGN
void CLASS_SIGN::OverridePin()
{
    if(m_Override.Pin.Enable)
        m_Pin = m_Override.Pin.data;
}

TEMPLATE_SIGN
void CLASS_SIGN::SetPinOverride(const std::vector<float>& _Pin)
{
    m_Override.Pin.data = _Pin;
    m_Override.Pin.Enable = true;
}


TEMPLATE_SIGN
void CLASS_SIGN::OverridePout()
{
    if(m_Override.Pout.Enable)
        m_Pout = m_Override.Pout.data;
}

TEMPLATE_SIGN
void CLASS_SIGN::SetPoutOverride(const std::vector<float>& _Pout)
{
    m_Override.Pout.data = _Pout;
    m_Override.Pout.Enable = true;
}



TEMPLATE_SIGN
void CLASS_SIGN::ComputeL()
{
    MacroAssert(m_Pin.size() == m_Pout.size());
    thrust::fill(m_L.begin(), m_L.end(), 0.0f);
    const double Ain_inv = 1.0 / m_Ain;
    const double Aout_inv = 1.0 / m_Aout;

    thrust::transform(m_Pin.begin(), m_Pin.end(), m_Pout.begin(),
        m_L.begin(), L_functor<float>(float(eps), float(Ain_inv), float(Aout_inv)));

#if 0 // this is not correct
    std::vector<float> binom{ 0.1, 0.2, 0.4, 0.2, 0.1 };  //size-5
    for(int i=0; i < 3; ++i)
        m_L = my_conv(binom, m_L, false);
#endif
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
double CLASS_SIGN::_Compute_V_1st_Term() const
{
    MacroAssert(m_Pin.size() == m_Pout.size());
    V_1st_term_functor<float> v_functor;
    thrust::plus<float> plus_float;
    double B = (double)thrust::inner_product(
        m_Pin.begin(),
        m_Pin.end(),
        m_Pout.begin(),
        float(0.0f), 
        plus_float, v_functor);

    double areaTerm = 1.0 / m_Ain - 1.0 / m_Aout;
    double ret = 0.5 * B * areaTerm;
    return ret;
}


TEMPLATE_SIGN
void CLASS_SIGN::ComputeV()
{
    const float firstTerm = (float) _Compute_V_1st_Term();
    MacroAssert(m_V.size() == m_points.size());

    // TODO: initialize m_V, use fill
#if 1
    //thrust::replace_if(m_V.begin(), m_V.end(), float_not_zero(), 0.0f);
    //thrust::transform_if(m_V.begin(), m_V.end(), m_V.begin(), [](float) {return 0.0f; }, float_not_zero());
    thrust::fill(m_V.begin(), m_V.end(), 0.0f);
    //V_2nd_term_functor vfunctor(THRUST_2_RAW(m_L), 0.0f, histSize);
#endif

    dim3 histSize((uint)m_Param.HistSize(), (uint)m_Param.HistSize(), (uint)m_Param.HistSize());
    V_2nd_term_functor vfunctor(THRUST_2_RAW(m_L), firstTerm, histSize);
    CudaStencil_NarrowBand<float> stencil_functor(float(m_Param.NarrowBand()));
    thrust::transform_if( m_points.begin(), m_points.end(), m_phi.begin(), m_V.begin(), vfunctor, stencil_functor);
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
void CLASS_SIGN::GetPhi(std::vector<float>& _phi) const
{
    _phi.resize(m_phi.size());
    thrust::copy(m_phi.begin(), m_phi.end(), _phi.begin());
}

TEMPLATE_SIGN
void CLASS_SIGN::SetPhi(const std::vector<float>& _phi)
{
    m_phi = _phi;
}
TEMPLATE_SIGN
void CLASS_SIGN::SetBlanked(const std::vector<char>& blanked)
{
    m_blanked.resize(blanked.size());
    thrust::copy(blanked.begin(), blanked.end(), m_blanked.begin());

    thrust::host_vector<char> host_blanked = blanked;
    thrust::host_vector<_Attrib> host_points = m_points;
    thrust::host_vector<_Attrib> host_included_points(host_points.size());
    auto endIter = host_included_points.end();
    try {
        endIter = thrust::copy_if(host_points.begin(), host_points.end(), host_blanked.begin(), host_included_points.begin(), is_not_blanked<char,bool>());
    }
    catch (std::exception e)
    {
        MacroWarning(e.what());
    }
    host_included_points.resize(endIter - host_included_points.begin());
    m_included_points = host_included_points;
    //_SetBlanking(m_points, m_included_points);
}

TEMPLATE_SIGN
void CLASS_SIGN::GetV(std::vector<float>& V) const
{
    V.resize(m_V.size());
    thrust::copy(m_V.begin(), m_V.end(), V.begin());
}

TEMPLATE_SIGN
void CLASS_SIGN::Sussman()
{
    dim3 dimBlock(8, 8, 8);
    //dim3 dimBlock(8, 8, 16);
    if (m_extent.depth == 1)
        dimBlock.z = 1;

    dim3 dimGrid((uint(m_extent.width) + dimBlock.x - 1) / dimBlock.x,
        (uint(m_extent.height) + dimBlock.y - 1) / dimBlock.y,
        (uint(m_extent.depth)  + dimBlock.z - 1) / dimBlock.z);

    // Copy m_phi to 2D/3D texture object
    m_TexPhi->LoadData(m_phi);

    int3 texDim{ (int)m_extent.width, (int)m_extent.height, (int)m_extent.depth };

    // Run Sussman Kernel and get back output in the phi device vector.
    _ClearUpdateMarkers();
    SussmanKernel <<<dimGrid, dimBlock>>> (THRUST_2_RAW(m_phi), THRUST_2_RAW(m_updated), m_phi.size(), 0.5f, m_TexPhi->TexObj(), texDim);
    checkCudaErrors(cudaDeviceSynchronize());
}


TEMPLATE_SIGN
void CLASS_SIGN::Compute_Max_e()
{
    auto E_range = thrust::minmax_element(m_V.begin(), m_V.end());
    //m_Max_e = (float) std::max(std::abs(*E_range.first), std::abs(*E_range.second));
    float high = *E_range.first;
    float low = *E_range.second;
    m_Max_e = std::max(std::abs(high), std::abs(low));
    /*
    m_Max_e = thrust::reduce(m_V.begin(), m_V.end(), 0, abs_max<float>());
    */
}

TEMPLATE_SIGN
void CLASS_SIGN::ProcessDivergence()
{
    _ClearUpdateMarkers();

    int threadsPerBlock = 1024;
    int numOfBlocks =(int(m_points.size()) + threadsPerBlock - 1) / threadsPerBlock;
    //printf("UpdatePhiKernel launch with %d blocks of %d threads\n", numOfBlocks, threadsPerBlock);

    // Run Update Phi Kernel and get back output in the phi device vector.
    const float3 params{float(m_Param.NarrowBand()), float(m_Param.StepSize()),float(m_Max_e)};
    UpdatePhiKernel << <numOfBlocks, threadsPerBlock >> > (
        THRUST_2_RAW(m_phi),
        m_phi.size(),
        THRUST_2_RAW(m_V),
        THRUST_2_RAW(m_updated),
        THRUST_2_RAW(m_blanked),
        params );

    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
void CLASS_SIGN::_ClearUpdateMarkers()
{
    signed char fill_char = 0;
    thrust::fill(m_updated.begin(), m_updated.end(), fill_char);
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
void CLASS_SIGN::_Histogram(thrust::device_vector<float>& fg, thrust::device_vector<float>& bg)
{
    //MacroFatalError("This function shouldn't be called.");
    dense_histogram(fg, m_Pin, _HistSize());
    dense_histogram(bg, m_Pout, _HistSize());
    m_Hin  = m_Pin;
    m_Hout = m_Pout;
}

TEMPLATE_SIGN
void CLASS_SIGN::_Histogram(thrust::device_vector<float2>& fg, thrust::device_vector<float2>& bg)
{
    int threadsPerBlock = 1024;
    if (!fg.empty())
    {
        int numOfBlocks = (int(fg.size()) + threadsPerBlock - 1) / threadsPerBlock;
        Histogram2D_kernel << <numOfBlocks, threadsPerBlock >> > (THRUST_2_RAW(fg), fg.size(), THRUST_2_RAW(m_Hin), m_Param.HistSize());
    }
    if (!bg.empty())
    {
        int numOfBlocks = (int(bg.size()) + threadsPerBlock - 1) / threadsPerBlock;
        Histogram2D_kernel << <numOfBlocks, threadsPerBlock >> > (THRUST_2_RAW(bg), bg.size(), THRUST_2_RAW(m_Hout), m_Param.HistSize());
    }
    checkCudaErrors(cudaDeviceSynchronize());
    /*
    auto total = thrust::reduce(m_Hin.begin(), m_Hin.end(), 0.0f, thrust::plus<float>());
    assert_equality(size_t(total), fg.size());
    total = thrust::reduce(m_Hout.begin(), m_Hout.end(), 0.0f, thrust::plus<float>());
    assert_equality(size_t(total), bg.size());
    */
}

TEMPLATE_SIGN
void CLASS_SIGN::_Histogram(thrust::device_vector<float3>& fg, thrust::device_vector<float3>& bg)
{
    int threadsPerBlock = 1024;
    if (!fg.empty())
    {
        int numOfBlocks = (int(fg.size()) + threadsPerBlock - 1) / threadsPerBlock;
        Histogram3D_kernel << <numOfBlocks, threadsPerBlock >> > (THRUST_2_RAW(fg), fg.size(), THRUST_2_RAW(m_Hin), m_Param.HistSize());
    }
    if (!bg.empty())
    {
        int numOfBlocks = (int(bg.size()) + threadsPerBlock - 1) / threadsPerBlock;
        Histogram3D_kernel << <numOfBlocks, threadsPerBlock >> > (THRUST_2_RAW(bg), bg.size(), THRUST_2_RAW(m_Hout), m_Param.HistSize());
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
dim3 CLASS_SIGN::GetBlockDimensions(const cudaExtent& extent, const dim3& threads)
{
    return dim3( uiDivUp(uint(extent.width), threads.x),
                 uiDivUp(uint(extent.height), threads.y),
                 uiDivUp(uint(extent.depth), threads.z) );
}


/*
TEMPLATE_SIGN
void CLASS_SIGN::_Convolve2(thrust::device_vector<float>& P_d, int iterations)
{
    if (_Dim == 1) MacroFatalError("This function should not be called.");

    vtkNew<vtkImageData> img;
    int dim[] = { (int)m_Param.HistSize(), (int)m_Param.HistSize(), (int)m_Param.HistSize() };
    if (_Dim == 2) dim[2] = 1;
    img->SetDimensions(dim);
    img->SetOrigin(0, 0, 0);
    img->SetSpacing(1, 1, 1);
    img->AllocateScalars(VTK_FLOAT, 1);
    std::vector<float> P_h(P_d.size());
    thrust::copy(P_d.begin(), P_d.end(), P_h.begin());
    std::memcpy(img->GetScalarPointer(), P_h.data(), sizeof(float)*P_h.size());

    vtkNew<vtkImageGaussianSmooth> gauss;
    gauss->SetInputData(img);
    gauss->SetDimensionality(_Dim);
    gauss->SetStandardDeviations(2.5, 2.5, 2.5);
    gauss->SetRadiusFactors(4,4,4);
    gauss->SetNumberOfThreads(m_Param.NumOfThreads());
    gauss->Update();

    std::memcpy(P_h.data(), gauss->GetOutput()->GetScalarPointer(), sizeof(float)*P_h.size());
    my_norm(P_h);
    thrust::copy(P_h.begin(), P_h.end(), P_d.begin());
}
*/

TEMPLATE_SIGN
void CLASS_SIGN::Convolve_CUDA(thrust::device_vector<float>& P, int iterations)
{
    /*StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);*/

    dim3 threads(8, 8, 8);
	if (_Dim == 2)
	{
		threads.x = 32;
		threads.y = 32;
		threads.z = 1;
	}
	else if (_Dim == 1)
	{
		threads.x = 1024;
		threads.y = 1;
		threads.z = 1;
	}

    cudaExtent extent = m_Histogram->GetExtent();
    dim3 blocks = GetBlockDimensions(extent, threads);
    if (_Dim < 3)      blocks.z = 1;
    else if (_Dim < 2) blocks.y = 1;

    int3 texDim{ (int)extent.width, (int)extent.height, (int)extent.depth };
    // X-convolve
    m_Histogram->LoadData(P);
    float3 mode = make_float3(1.0f, 0.0f, 0.0f);
    //sdkStartTimer(&hTimer);
	//std::cout << "BlockDim   = " << blocks.x << " " << blocks.y << " " << blocks.z << "\n";
	//std::cout << "threadsDim = " << threads.x << " " << threads.y << " " << threads.z << "\n";
    ConvolveKernel <<<blocks, threads>>> (THRUST_2_RAW(P), P.size(), mode, m_Histogram->TexObj(), texDim);
    checkCudaErrors(cudaDeviceSynchronize());
    /*sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
    std::cout << "gpuTIME = " << gpuTime << std::endl;*/

    if (_Dim > 1)
    {
        // Y-convolve
        m_Histogram->LoadData(P);
        mode = make_float3(0.0f, 1.0f, 0.0f);
        ConvolveKernel << <blocks, threads >> > (THRUST_2_RAW(P), P.size(), mode, m_Histogram->TexObj(), texDim);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if (_Dim > 2)
    {
        // Z-convolve
        m_Histogram->LoadData(P);
        mode = make_float3(0.0f, 0.0f, 1.0f);
        ConvolveKernel <<<blocks, threads>>> (THRUST_2_RAW(P), P.size(), mode, m_Histogram->TexObj(), texDim);
        checkCudaErrors(cudaDeviceSynchronize());
    }

	/*
    // re-normalize the result
    float sum = thrust::reduce(P.begin(), P.end(), 0.0f);

	// Trial sum kernel
	SumKernel <<<blocks, threads>>> (THRUST_2_RAW(m_buffer), THRUST_2_RAW(P), P.size());
	float trial_sum = m_buffer[0];
	std::cout << "thrust::reduce = " << sum << ", SumKernel = " << trial_sum << std::endl;

    //thrust::transform(P.begin(), P.end(), P.begin(), [sum]__device__(auto p) { return float(p) / sum; });
    thrust::transform(P.begin(), P.end(), P.begin(), CudaFunctor_Scale<float>(1.0f / sum));
	*/
}

TEMPLATE_SIGN
void CLASS_SIGN::UpdateFromLeft(const void* srcPhi, const int srcGPUId)
{
    if (srcPhi)
    {
        MacroConfirm(srcGPUId >= 0);
        //size_t sizeOfZSlice = m_extent.height * m_extent.width;
        auto ret = cudaMemcpyPeer(THRUST_2_RAW(m_phi), m_GPUDeviceId, srcPhi, srcGPUId, _GhostSize() * sizeof(float));
        checkCudaErrors(ret);
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::UpdateFromRight(const void* srcPhi, const int srcGPUId)
{
    if (srcPhi)
    {
        MacroConfirm(srcGPUId >= 0);

        auto ret = cudaMemcpyPeer(
            static_cast<void*>(static_cast<float*>(THRUST_2_RAW(m_phi)) + m_phi.size() - _GhostSize()),
            m_GPUDeviceId,
            srcPhi,
            srcGPUId,
            _GhostSize() * sizeof(float));

        checkCudaErrors(ret);
    }
}

/// Get non-ghost layer on the "left" side (Begin side of the array).
TEMPLATE_SIGN
const void* CLASS_SIGN::GetLeftLayer() const
{
    return (static_cast<const float*>(THRUST_2_RAW(m_phi)) + _GhostSize());
}

/// Get non-ghost layer on the "right" side (End side of the array).
TEMPLATE_SIGN
const void* CLASS_SIGN::GetRightLayer() const
{
    return (static_cast<const float*>(THRUST_2_RAW(m_phi)) + m_phi.size() - 2*_GhostSize());
}

TEMPLATE_SIGN
const void* CLASS_SIGN::GetHinDevicePtr() const
{
    return (THRUST_2_RAW(m_Hin));
}

TEMPLATE_SIGN
const void* CLASS_SIGN::GetHoutDevicePtr() const
{
    return (THRUST_2_RAW(m_Hout));
}

TEMPLATE_SIGN
void CLASS_SIGN::Update_H(const float* Hin, const float* Hout, int srcGPUId)
{
    // TODO:: make sure you are not adding on top of old expired histogram
    auto ret = cudaMemcpyPeer(THRUST_2_RAW(m_Peer_Hin), m_GPUDeviceId, Hin, srcGPUId, _HistSize() * sizeof(float));
    checkCudaErrors(ret);
    thrust::transform(m_Hin.begin(), m_Hin.end(), m_Peer_Hin.begin(), m_Hin.begin(), thrust::plus<float>());

    ret = cudaMemcpyPeer(THRUST_2_RAW(m_Peer_Hout), m_GPUDeviceId, Hout, srcGPUId, _HistSize() * sizeof(float));
    checkCudaErrors(ret);
    thrust::transform(m_Hout.begin(), m_Hout.end(), m_Peer_Hout.begin(), m_Hout.begin(), thrust::plus<float>());

    checkCudaErrors(cudaDeviceSynchronize());
}

TEMPLATE_SIGN
void CLASS_SIGN::SetGlobalHin(const void* srcHin, const int srcGPUId)
{
    auto ret = cudaMemcpyPeer(THRUST_2_RAW(m_Hin), m_GPUDeviceId, srcHin, srcGPUId, _HistSize() * sizeof(float));
    checkCudaErrors(ret);
}

TEMPLATE_SIGN
void CLASS_SIGN::SetGlobalHout(const void* srcHout, const int srcGPUId)
{
    auto ret = cudaMemcpyPeer(THRUST_2_RAW(m_Hout), m_GPUDeviceId, srcHout, srcGPUId, _HistSize() * sizeof(float));
    checkCudaErrors(ret);
}

TEMPLATE_SIGN
template<typename T>
void CLASS_SIGN::_SetBlanking(const thrust::device_vector<T>& input, thrust::device_vector<T>& output)
{
    output.resize(input.size());
    checkCudaErrors(cudaDeviceSynchronize());
#if 0
    thrust::copy(input.begin(), input.end(), output.begin());
#else
    auto endIter = output.end();
    try {
        endIter = thrust::copy_if(input.begin(), input.end(), m_blanked.begin(), output.begin(), is_not_blanked<int,bool>());
    }
    catch (std::exception e)
    {
        MacroPrint(input.size());
        MacroPrint(output.size());
        MacroPrint(m_blanked.size());
        MacroWarning(e.what());
    }
    output.resize(endIter - output.begin());
#endif
}  

inline void copy_attrib(float& dst, const float& src)
{
    dst = src;
}
inline void copy_attrib(float2& dst, const float2& src)
{
    dst.x = src.x;
    dst.y = src.y;
}
inline void copy_attrib(float3& dst, const float3& src)
{
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
}

TEMPLATE_SIGN
template<typename T>
std::vector<T> CLASS_SIGN::Get_StdVector(const thrust::device_vector<T>& Input)
{
    std::vector<T> ret(Input.size());
    for (size_t i = 0; i < Input.size(); ++i)
    {
        copy_attrib(ret[i], (T)Input[i]);
    }

    return ret;
}

TEMPLATE_SIGN
const std::vector<float> CLASS_SIGN::GetHin_Std()  const
{
    return Get_StdVector(m_Hin);
}

TEMPLATE_SIGN
const std::vector<float> CLASS_SIGN::GetHout_Std()  const
{
    return Get_StdVector(m_Hout);
}



template class Bhat::FeatureSpaceCuda<1, float>;
template class Bhat::FeatureSpaceCuda<2, float2>;
template class Bhat::FeatureSpaceCuda<3, float3>;


