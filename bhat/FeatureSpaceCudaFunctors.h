#pragma once
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

template<typename T> struct CudaFunctor_Scale
{
    const T factor;
    CudaFunctor_Scale(const T _factor) : factor(_factor)
    { }
    __device__ T operator()(const T& value)
    {
        return (value * factor);
    }
};

template<typename T> struct CudaFunctor_NarrowBand
{
    const T stepSize, max_e, eps, del, pi;
    CudaFunctor_NarrowBand(const T _stepSize, const T _max_e, const T _eps, const T _del, const T _pi)
        : stepSize(_stepSize), max_e(_max_e), eps(_eps), del(_del), pi(_pi)
    {
    }

    __device__ T operator()(const T& phi, const T& E)
    {
        T ret = phi;
        //T delta_phi = 0.5 * (1.0 + cosf(pi * phi / del));
        //ret = phi + stepSize * delta_phi  * E;
        //ret = phi + delta_phi*(stepSize / (max_e + eps)) * E;
        ret = phi + stepSize / (max_e + eps) * E;
        return ret;
    }
};


template<typename T> struct CudaStencil_NarrowBand
{
    const T band;
    CudaStencil_NarrowBand(const T& _band) : band(_band)
    {
    }

    __device__ bool operator()(const T& curr_value)
    {
        if (fabsf(curr_value) <= band || band < 0)
            return true;
        else
            return false;
    }
};

template<typename _InType, typename _OutType> struct CudaStencil_Positive
{
    __device__ _OutType operator()(const _InType& curr_value)
    {
        if (curr_value > 0)
            return _OutType(1);
        else
            return _OutType(0);
    }
};

template<typename _InType, typename _OutType> struct CudaStencil_NonNegative
{
    __device__ _OutType operator()(const _InType& curr_value)
    {
        //if (curr_value >= 0 && curr_value < 10)
        if (curr_value >= 0)
            return _OutType(1);
        else
            return _OutType(0);
    }
};


template<typename _InType, typename _OutType> struct CudaStencil_Negative
{
    __device__ _OutType operator()(const _InType& curr_value)
    {
        if (curr_value < 0)
            return _OutType(1);
        else
            return _OutType(0);
    }
};

template<typename T>
struct normalize_functor {
    const T area;
    normalize_functor(T _area) : area(_area) {}
    /*__host__*/ __device__ T operator()(T x) {
        if (x == 0) x = 1;
        return x / area;
    }
};

template<typename T> struct L_functor {
    const T eps, Ain_inv, Aout_inv;
    L_functor(T _eps, T _Ain_inv, T _Aout_inv) : eps(_eps), Ain_inv(_Ain_inv), Aout_inv(_Aout_inv)
    {
    }
    /*__host__*/ __device__ T operator()(const T& Pin, const T& Pout)
    {
        T ret = + Aout_inv * sqrtf(Pin  / (eps + Pout))
                - Ain_inv  * sqrtf(Pout / (eps +  Pin));
        return ret;
    }
};

template<typename T> struct V_1st_term_functor {
    /*__host__*/ __device__ T operator()(const T& Pin, const T& Pout)
    {
        return sqrtf(Pin * Pout);
    }
};

struct V_2nd_term_functor
{
    const float* L;
    const float firstTerm;
    const dim3 histSize;
    V_2nd_term_functor(const float* _L, const float _firstTerm, dim3 _histSize)
        : L(_L), firstTerm(_firstTerm), histSize(_histSize)
    { }

#if 1
    __device__ float operator()(const float& p)
    {
        int idx = int(p);
        float secondTerm = L[idx];
        return (firstTerm + 0.5f * secondTerm);
    }

    __device__ float operator()(const float2& p)
    {
        int idx = int(p.x);
        int idy = int(p.y);
        float secondTerm = L[idx + histSize.x*idy];
        return (firstTerm + 0.5f * secondTerm);
    }

    __device__ float operator()(const float3& p)
    {
        int idx = int(p.x);
        int idy = int(p.y);
        int idz = int(p.z);
        float secondTerm = L[idx + histSize.x*idy + histSize.x*histSize.y*idz];
        return (firstTerm + 0.5f * secondTerm);
    }

#else
    __device__ T operator()(const T& x)
    {
        //float* h = new float[size_t(histSize)];
        float h[256];
        float sum = 0.0f;
        for (uint i = 0; i < histSize; ++i)
        {
            auto& he = h[i];
            he = (0.5f / sigma)*(1.0f + cosf(3.142f*(i - x) / sigma));
            he = (he <= sigma && he >= -sigma) ? he : 0.0f;
            sum += he;
        }

        float secondTerm = 0.0f;
        for (uint i = 0; i < histSize; ++i)
        {
            auto& he = h[i];
            he /= sum;
            secondTerm += he * L[i];
        }

        //delete h; 
        return (firstTerm + 0.5f * secondTerm);
    }
#endif
};

template<typename T> struct CudaFunctor_mask {
    __device__ bool operator()(const T& phi) {
        if (phi <= 0)
            return true;
        else
            return false;
    }
};

template<typename T> struct CudaFunctor_ak_minus_V {
    const bool approx = true;
    CudaFunctor_ak_minus_V()
    { }

    __device__ T operator()(const T& ak, const T& V)
    {
        if (approx)
            return(ak - copysignf(0.25f,V));
        else
            return (ak - V);
    }
};

struct is_true {
    /*__host__*/ __device__ bool operator()(const bool x) {
        return x;
    }
};
struct is_false {
    /*__host__*/ __device__ bool operator()(const bool x) {
        return !x;
    }
};
struct float_not_zero {
    /*__host__*/ __device__ bool operator()(const float x) {
        return (x < -1e-6 || x > 1e-6);
    }
};

/*
template<typename _InType, typename _OutType> struct CudaStencil_Negative
{
    __device__ _OutType operator()(const _InType& curr_value)
    {
        if (curr_value < 0)
            return _OutType(1);
        else
            return _OutType(0);
    }
};
*/
template<typename _InType, typename _OutType> 
struct is_not_blanked {
    __host__ __device__ _OutType operator()(const _InType& curr_value)
    {
        if (curr_value == 0)
            return _OutType(1);
        else
            return _OutType(0);
    }
};

struct is_ghost {
    /*__host__*/ __device__ bool operator()(const int x) {
        return bool(x == 2);
    }
};

struct  stencil_phi{
    __host__ __device__ float operator()(const float x, const bool stencilValue)
    {
        if (!stencilValue) return 1024.0f;
        else return x;
    }
};

template<typename _InOutType> 
struct abs_max {
    __host__ __device__ _InOutType operator()(_InOutType A, _InOutType B )
    {
        A = fabsf(A);
        B = fabsf(B);
        return (A > B ? A : B);
    }
};

