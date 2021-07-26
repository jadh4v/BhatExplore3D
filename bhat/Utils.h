#pragma once

#define THRUST_2_RAW(x) thrust::raw_pointer_cast(x.data())
//Round a / b to nearest higher integer value
inline int uiDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

template<typename T>
static std::vector<T> device_to_host(const thrust::device_vector<T>& dev)
{
    std::vector<T> ret(dev.size());
    thrust::copy(dev.begin(), dev.end(), ret.begin());
    return ret;
}

template<typename T>
static thrust::device_vector<T> host_to_device(const std::vector<T>& dev)
{
    thrust::device_vector<T> ret(dev.size());
    thrust::copy(dev.begin(), dev.end(), ret.begin());
    return ret;
}

template<typename T>
inline void assert_equality(const T& value1, const T& value2) { \
    if (value1 != value2) \
        MacroWarning("Equality test failed. " << value1 << " == " << value2); \
}

inline __device__ __host__ size_t attrib_to_index(const float p, const size_t hsz)
{
    return size_t(p);
}
inline __device__ __host__ size_t attrib_to_index(const float2& p, const size_t hsz)
{
    return size_t(p.x + p.y*hsz);
}
inline __device__ __host__ size_t attrib_to_index(const float3& p, const size_t hsz)
{
    return size_t(p.x + p.y*hsz + p.z*hsz*hsz);
}

/*
template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

//#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
*/